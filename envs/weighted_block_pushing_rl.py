"""Gym environment for block pushing tasks (2D Shapes and 3D Cubes)."""

import numpy as np

import gym
from collections import OrderedDict
from dataclasses import dataclass
from gym import spaces
from gym.utils import seeding

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image

import skimage

from cswm import utils


mpl.use('Agg')

def diamond(r0, c0, width, im_size):
    rr, cc = [r0, r0 + width // 2, r0 + width, r0 + width // 2], [c0 + width // 2, c0, c0 + width // 2, c0 + width]
    return skimage.draw.polygon(rr, cc, im_size)

def square(r0, c0, width, im_size):
    rr, cc = [r0, r0 + width, r0 + width, r0], [c0, c0, c0 + width, c0 + width]
    return skimage.draw.polygon(rr, cc, im_size)

def triangle(r0, c0, width, im_size):
    rr, cc = [r0, r0 + width, r0 + width], [c0 + width//2, c0, c0 + width]
    return skimage.draw.polygon(rr, cc, im_size)

def cross(r0, c0, width, im_size):
    diff1 = width // 3 + 1
    diff2 = 2 * width // 3
    rr = [r0 + diff1, r0 + diff2, r0 + diff2, r0 + width, r0 + width,
            r0 + diff2, r0 + diff2, r0 + diff1, r0 + diff1, r0, r0, r0 + diff1]
    cc = [c0, c0, c0 + diff1, c0 + diff1, c0 + diff2, c0 + diff2, c0 + width,
            c0 + width, c0 + diff2, c0 + diff2, c0 + diff1, c0 + diff1]
    return skimage.draw.polygon(rr, cc, im_size)

def pentagon(r0, c0, width, im_size):
    diff1 = width // 3 - 1
    diff2 = 2 * width // 3 + 1
    rr = [r0 + width // 2, r0 + width, r0 + width, r0 + width // 2, r0]
    cc = [c0, c0 + diff1, c0 + diff2, c0 + width, c0 + width // 2]
    return skimage.draw.polygon(rr, cc, im_size)

def parallelogram(r0, c0, width, im_size):
    rr, cc = [r0, r0 + width, r0 + width, r0], [c0, c0 + width // 2, c0 + width, c0 + width - width // 2]
    return skimage.draw.polygon(rr, cc, im_size)


def scalene_triangle(r0, c0, width, im_size):
    rr, cc = [r0, r0 + width, r0 + width//2], [c0 + width - width // 2, c0, c0 + width]
    return skimage.draw.polygon(rr, cc, im_size)

def fig2rgb_array(fig):
    fig.canvas.draw()
    buffer = fig.canvas.tostring_rgb()
    width, height = fig.canvas.get_width_height()
    return np.fromstring(buffer, dtype=np.uint8).reshape(height, width, 3)


def render_cubes(objects, width):
    voxels = np.zeros((width, width, width), dtype=np.bool)
    colors = np.empty(voxels.shape, dtype=object)

    cols = ['purple', 'green', 'orange', 'blue', 'brown']

    for i, pos in objects.items():
        voxels[pos[0], pos[1], 0] = True
        colors[pos[0], pos[1], 0] = cols[i]

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.w_zaxis.set_pane_color((0.5, 0.5, 0.5, 1.0))
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.line.set_lw(0.)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.voxels(voxels, facecolors=colors, edgecolor='k')

    im = fig2rgb_array(fig)
    plt.close(fig)
    im = np.array(  # Crop and resize
        Image.fromarray(im[215:455, 80:570]).resize((50, 50), Image.ANTIALIAS))
    return im / 255.


@dataclass
class Coord:
    x: int
    y: int

    def __add__(self, other):
        return Coord(self.x + other.x,
                     self.y + other.y)


@dataclass
class Object:
    pos: Coord
    weight: int


class InvalidMove(BaseException):
    pass


class InvalidPush(BaseException):
    pass


class BlockPushingRL(gym.Env):
    """Gym environment for block pushing task."""

    def __init__(self, width=5, height=5, render_type='cubes',
                 *, num_objects=5, mode='Train', cmap='Blues', typ='Observed',
                 num_weights=None, random_colors=False, seed=None):
        self.width = width
        self.height = height
        self.render_type = render_type
        self.mode = mode
        self.cmap = cmap
        self.typ = typ
        self.new_colors = None

        if typ in ['Unobserved', 'FixedUnobserved'] and "FewShot" in mode:
            self.n_f = int(mode[-1])
            if cmap == 'Sets':
                self.new_colors = np.random.choice(12, self.n_f, replace=False)
            elif cmap == 'Pastels':
                self.new_colors = np.random.choice(8, self.n_f, replace=False)
            else:
                print("something went wrong")

        self.num_objects = num_objects
        self.num_actions = 5 * self.num_objects  # Move StayNESW
        if num_weights is None:
            num_weights = num_objects
        self.num_weights = num_weights

        self.np_random = None
        self.game = None

        # Initialize to pos outside of env for easier collision resolution.
        self.objects = OrderedDict()

        # If True, then check for collisions and don't allow two
        #   objects to occupy the same position.
        self.collisions = True

        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(3, self.width, self.height),
            dtype=np.float32
        )

        self.seed(seed)
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render_grid(self):
        im = np.zeros((3, self.width, self.height))
        for idx, obj in self.objects.items():
            im[:, obj.pos.x, obj.pos.y] = self.colors[idx][:3]
        return im

    def render_circles(self):
        im = np.zeros((self.width * 10, self.height * 10, 3), dtype=np.float32)
        for idx, obj in self.objects.items():
            rr, cc = skimage.draw.circle(
                obj.pos.x * 10 + 5, obj.pos.y * 10 + 5, 5, im.shape)
            im[rr, cc, :] = self.colors[idx][:3]
        return im.transpose([2, 0, 1])

    def render_shapes(self):
        im = np.zeros((self.width * 10, self.height * 10, 3), dtype=np.float32)
        for idx, obj in self.objects.items():
            if self.shapes[idx] == 0:
                rr, cc = skimage.draw.circle(
                    obj.pos.x * 10 + 5, obj.pos.y * 10 + 5, 5, im.shape)
            elif self.shapes[idx] == 1:
                rr, cc = triangle(
                    obj.pos.x * 10, obj.pos.y * 10, 10, im.shape)
            elif self.shapes[idx] == 2:
                rr, cc = square(
                    obj.pos.x * 10, obj.pos.y * 10, 10, im.shape)
            elif self.shapes[idx] == 3:
                rr, cc = diamond(
                    obj.pos.x * 10, obj.pos.y * 10, 10, im.shape)
            elif self.shapes[idx] == 4:
                rr, cc = cross(
                    obj.pos.x * 10, obj.pos.y * 10, 10, im.shape)
            elif self.shapes[idx] == 5:
                rr, cc = pentagon(
                    obj.pos.x * 10, obj.pos.y * 10, 10, im.shape)
            elif self.shapes[idx] == 6:
                rr, cc = parallelogram(
                    obj.pos.x * 10, obj.pos.y * 10, 10, im.shape)
                im[rr, cc, :] = self.colors[obj.color][:3]
            else:
                rr, cc = scalene_triangle(
                    obj.pos.x * 10, obj.pos.y * 10, 10, im.shape)
                im[rr, cc, :] = self.colors[obj.color][:3]

            im[rr, cc, :] = self.colors[idx][:3]

        return im.transpose([2, 0, 1])

    def render_cubes(self):
        im = render_cubes(self.objects, self.width)
        return im.transpose([2, 0, 1])

    def render(self):
        return dict(
            grid=self.render_grid,
            circles=self.render_circles,
            shapes=self.render_shapes,
            cubes=self.render_cubes,
        )[self.render_type]()

    def get_state(self):
        im = np.zeros(
            (self.num_objects, self.width, self.height), dtype=np.int32)
        for idx, obj in self.objects.items():
            im[idx, obj.pos.x, obj.pos.y] = 1
        return im

    def get_sparse_reward(self, target_state):
        return np.sum(np.minimum(state, self.target[0])) / self.num_objects

    def get_dense_reward(self, target_objects):
        distance = 0.0
        for i in range(self.num_objects):
            distance += np.abs(self.objects[i].pos.x - target_objects[i].pos.x) +\
                        np.abs(self.objects[i].pos.y - target_objects[i].pos.y)

        distance /= ((self.height - 1) * (self.width - 1) * self.num_objects)
        return -distance

    def reset(self):
        if self.typ == 'FixedUnobserved':
            self.shapes = np.arange(self.num_objects)
        elif self.mode == 'ZeroShotShape':
            self.shapes = np.random.choice(6, self.num_objects)
        else:
            self.shapes = np.random.choice(3, self.num_objects)

        self.objects = OrderedDict()
        if self.typ == 'Observed':
            self.colors, weights = utils.get_colors_and_weights(
                cmap=self.cmap,
                num_colors=self.num_objects,
                observed=True,
                mode=self.mode)
        else:
            self.colors, weights = utils.get_colors_and_weights(
                cmap=self.cmap,
                num_colors=self.num_objects,
                observed=False,
                mode=self.mode,
                new_colors=self.new_colors)

        # Randomize object position.
        while len(self.objects) < self.num_objects:
            idx = len(self.objects)
            # Re-sample to ensure objects don't fall on same spot.
            while not (idx in self.objects and
                       self.valid_pos(self.objects[idx].pos, idx)):
                self.objects[idx] = Object(
                    pos=Coord(
                        x=np.random.choice(np.arange(self.width)),
                        y=np.random.choice(np.arange(self.height)),
                    ),
                    weight=weights[idx])

        self.target = (np.zeros([self.num_objects, self.width, self.height], dtype=int),
            np.zeros([3, self.width * 10, self.height * 10]))
        self.target_objects = self.objects.copy()

        self.get_target()

        return (self.get_state(), self.render()), self.target

    def valid_pos(self, pos, obj_id):
        """Check if position is valid."""
        if pos.x not in range(0, self.width):
            return False
        if pos.y not in range(0, self.height):
            return False

        if self.collisions:
            for idx, obj in self.objects.items():
                if idx == obj_id:
                    continue

                if pos == obj.pos:
                    return False

        return True

    def valid_move(self, obj_id, offset: Coord):
        """Check if move is valid."""
        old_obj = self.objects[obj_id]
        new_pos = old_obj.pos + offset
        return self.valid_pos(new_pos, obj_id)

    def occupied(self, pos: Coord):
        for idx, obj in self.objects.items():
            if obj.pos == pos:
                return idx
        return None

    def translate(self, obj_id, offset: Coord, n_parents=0):
        """"Translate object pixel.

        Args:
            obj_id: ID of object.
            offset: (x, y) coordinate.
        """
        obj = self.objects[obj_id]

        other_object_id = self.occupied(obj.pos + offset)
        if other_object_id is not None:
            if n_parents == 1:
                # cannot push two objects
                raise InvalidPush()
            if obj.weight > self.objects[other_object_id].weight:
                self.translate(other_object_id, offset,
                               n_parents=n_parents+1)
            else:
                raise InvalidMove()
        if not self.valid_move(obj_id, offset):
            raise InvalidMove()

        self.objects[obj_id] = Object(
            pos=obj.pos+offset, weight=obj.weight)

    def step(self, action: int):
        directions = [Coord(0, 0),
                      Coord(-1, 0),
                      Coord(0, 1),
                      Coord(1, 0),
                      Coord(0, -1)]

        direction = action % 5
        obj_id = action // 5

        done = False
        info = {'invalid_push': False}
        try:
            self.translate(obj_id, directions[direction])
        except InvalidMove:
            pass
        except InvalidPush:
            info['invalid_push'] = True

        state = self.get_state()
        img = self.render()

        # reward = self.get_sparse_reward(self.target[0])
        reward = self.get_dense_reward(self.target_objects)

        state_obs = (state, img)
        return state_obs, reward, done, info

    def sample_step(self, action: int):
        directions = [Coord(0, 0),
                      Coord(-1, 0),
                      Coord(0, 1),
                      Coord(1, 0),
                      Coord(0, -1)]

        direction = action % 5
        obj_id = action // 5
        done = False
        info = {'invalid_push': False}

        objects = self.objects.copy()
        try:
            self.translate(obj_id, directions[direction])
        except InvalidMove:
            pass
        except InvalidPush:
            info['invalid_push'] = True

        reward = self.get_dense_reward(self.target_objects)
        next_obs = self.render()
        self.objects = objects

        return reward, next_obs

    def get_target(self, num_steps=10):
        objects = self.objects.copy()

        for i in range(num_steps):
            move = np.random.choice(self.num_objects * 5)
            state, _, _, _ = self.step(move)

        self.target_objects = self.objects.copy()
        self.target = state
        self.objects = objects
