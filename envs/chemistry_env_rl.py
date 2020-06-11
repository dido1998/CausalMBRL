"""Gym environment for chnging colors of shapes."""

import numpy as np
import torch
import torch.nn as nn

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

def random_dag(M, N, g = None):
    """Generate a random Directed Acyclic Graph (DAG) with a given number of nodes and edges."""
    if g is None:
        expParents = 5
        idx        = np.arange(M).astype(np.float32)[:,np.newaxis]
        idx_maxed  = np.minimum(idx, expParents)
        p          = np.broadcast_to(idx_maxed/(idx+1), (M, M))
        B          = np.random.binomial(1, p)
        B          = np.tril(B, -1)
        return B
    else:
        gammagt = np.zeros((M, M))            
        for e in g.split(","):
            if e == "": continue
            nodes = e.split("->")
            if len(nodes) <= 1: continue
            nodes = [int(n) for n in nodes]
            for src, dst in zip(nodes[:-1], nodes[1:]):
                if dst > src:
                    gammagt[dst,src] = 1
                elif dst == src:
                    raise ValueError("Edges are not allowed from " +
                                     str(src) + " to oneself!")
                else:
                    raise ValueError("Edges are not allowed from " +
                                     str(src) + " to ancestor " +
                                     str(dst) + " !")
        return gammagt



def square(r0, c0, width, im_size):
    rr, cc = [r0, r0 + width, r0 + width, r0], [c0, c0, c0 + width, c0 + width]
    return skimage.draw.polygon(rr, cc, im_size)


def triangle(r0, c0, width, im_size):
    rr, cc = [r0, r0 + width, r0 + width], [c0 + width//2, c0, c0 + width]
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


class MLP(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.layers = []
        for i in range(1, len(dims)):
            self.layers.append(nn.Linear(dims[i-1], dims[i]))
            torch.nn.init.orthogonal_(self.layers[-1].weight.data, 1.5)
            torch.nn.init.uniform_(self.layers[-1].bias.data, -1.1, +1.1)
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x, mask):
        x = x * mask
        for i, l in enumerate(self.layers):
            if i == len(self.layers) - 1:
                x = torch.softmax(l(x), dim = 1)
            else:
                x = torch.relu(l(x))
        x = torch.distributions.one_hot_categorical.OneHotCategorical(probs = x).sample()
        return x


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
    color: int


class ColorChangingRL(gym.Env):
    """Gym environment for block pushing task."""

    def __init__(self, width=5, height=5, render_type='cubes',
                 *, num_objects=5,
                 num_colors=None,  pal_id = 0, max_steps = 50, seed=None):
        np.random.seed(0)
        self.width = width
        self.height = height
        self.render_type = render_type

        self.num_objects = num_objects
        
        if num_colors is None:
            num_colors = num_objects
        self.num_colors = num_colors
        self.num_actions = self.num_objects * self.num_colors
        self.max_steps = max_steps
        self.cur_step = 0
        
        self.mlps = []
        self.mask = None

        self.pal_id = pal_id

        self.palletes= [['Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2', 'Set1'], 
                   ['Set2', 'Set3', 'tab10', 'tab20', 'tab20b', 'tab20c']]

        self.colors, _ = utils.get_colors_and_weights(cmap = self.palletes[self.pal_id][np.random.randint(0, len(self.palletes[self.pal_id]))], num_colors = self.num_colors)
        self.object_to_color = [torch.zeros(self.num_colors) for _ in range(self.num_objects)]
        self.object_to_color_target = [torch.zeros(self.num_colors) for _ in range(self.num_objects)]

        self.np_random = None
        self.game = None
        self.target = None

        self.adjacency_matrix = None

        self.colors, _ = utils.get_colors_and_weights(cmap = self.palletes[self.pal_id][np.random.randint(0, len(self.palletes[self.pal_id]))], num_colors = self.num_colors)

        mlp_dims = [self.num_objects * self.num_colors, 4 * self.num_objects * self.num_colors, self.num_colors]

        self.mlps = []

        for i in range(self.num_objects):
            self.mlps.append(MLP(mlp_dims))
        num_nodes = self.num_objects
        num_edges = np.random.randint(1, (((num_nodes) * (num_nodes - 1)) // 2) + 1)
        #if graph is None:
        self.adjacency_matrix = random_dag(num_nodes, num_edges)
        #else:
        #    self.adjacency_matrix = random_dag(num_nodes, num_nodes, g = graph)

        self.adjacency_matrix = torch.from_numpy(self.adjacency_matrix).float()

        

        # If True, then check for collisions and don't allow two
        #   objects to occupy the same position.
        self.collisions = True
        # Initialize to pos outside of env for easier collision resolution.
        self.objects = OrderedDict()
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
                    color=torch.argmax(self.object_to_color[idx]))

        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = (spaces.Box(
            low=0, high=1,
            shape=(3, self.width, self.height),
            dtype=np.float32
        ), spaces.Box(
            low=0, high=1,
            shape=(3, self.width, self.height),
            dtype=np.float32
        ))

        self.seed(seed)
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render_grid(self):
        im = np.zeros((3, self.width, self.height))
        for idx, obj in self.objects.items():
            im[:, obj.pos.x, obj.pos.y] = self.colors[obj.color][:3]
        return im

    def render_circles(self):
        im = np.zeros((self.width * 10, self.height * 10, 3), dtype=np.float32)
        for idx, obj in self.objects.items():
            rr, cc = skimage.draw.circle(
                obj.pos.x * 10 + 5, obj.pos.y * 10 + 5, 5, im.shape)
            im[rr, cc, :] = self.colors[obj.color][:3]
        return im.transpose([2, 0, 1])

    def render_shapes(self):
        im = np.zeros((self.width * 10, self.height * 10, 3), dtype=np.float32)
        for idx, obj in self.objects.items():
            if idx % 3 == 0:
                rr, cc = skimage.draw.circle(
                    obj.pos.x * 10 + 5, obj.pos.y * 10 + 5, 5, im.shape)
                im[rr, cc, :] = self.colors[obj.color][:3]
            elif idx % 3 == 1:
                rr, cc = triangle(
                    obj.pos.x * 10, obj.pos.y * 10, 10, im.shape)
                im[rr, cc, :] = self.colors[obj.color][:3]
            else:
                rr, cc = square(
                    obj.pos.x * 10, obj.pos.y * 10, 10, im.shape)
                im[rr, cc, :] = self.colors[obj.color][:3]
        return im.transpose([2, 0, 1])


    def render_grid_target(self):
        im = np.zeros((3, self.width, self.height))
        for idx, obj in self.objects.items():
            im[:, obj.pos.x, obj.pos.y] = self.colors[torch.argmax(self.object_to_color_target[idx])][:3]
        return im

    def render_circles_target(self):
        im = np.zeros((self.width * 10, self.height * 10, 3), dtype=np.float32)
        for idx, obj in self.objects.items():
            rr, cc = skimage.draw.circle(
                obj.pos.x * 10 + 5, obj.pos.y * 10 + 5, 5, im.shape)
            im[rr, cc, :] = self.colors[torch.argmax(self.object_to_color_target[idx])][:3]
        return im.transpose([2, 0, 1])

    def render_shapes_target(self):
        im = np.zeros((self.width * 10, self.height * 10, 3), dtype=np.float32)
        for idx, obj in self.objects.items():
            if idx % 3 == 0:
                rr, cc = skimage.draw.circle(
                    obj.pos.x * 10 + 5, obj.pos.y * 10 + 5, 5, im.shape)
                im[rr, cc, :] = self.colors[torch.argmax(self.object_to_color_target[idx])][:3]
            elif idx % 3 == 1:
                rr, cc = triangle(
                    obj.pos.x * 10, obj.pos.y * 10, 10, im.shape)
                im[rr, cc, :] = self.colors[torch.argmax(self.object_to_color_target[idx])][:3]
            else:
                rr, cc = square(
                    obj.pos.x * 10, obj.pos.y * 10, 10, im.shape)
                im[rr, cc, :] = self.colors[torch.argmax(self.object_to_color_target[idx])][:3]
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
        )[self.render_type](), dict(
            grid=self.render_grid_target,
            circles=self.render_circles_target,
            shapes=self.render_shapes_target,
            cubes=self.render_cubes,
        )[self.render_type]() 

    def get_state(self):
        im = np.zeros(
            (self.num_objects * self.num_colors, self.width, self.height), dtype=np.int32)
        im_target = np.zeros(
            (self.num_objects * self.num_colors, self.width, self.height), dtype=np.int32)
        for idx, obj in self.objects.items():
            im[idx * self.num_colors + obj.color, obj.pos.x, obj.pos.y] =  1
            im_target[idx * self.num_colors + torch.argmax(self.object_to_color_target[idx]), obj.pos.x, obj.pos.y] =  1
        return im, im_target

    def generate_masks(self):
        mask = self.adjacency_matrix.unsqueeze(-1)
        mask = mask.repeat(1, 1, self.num_colors)
        self.mask = mask.view(self.adjacency_matrix.size(0), -1)

    def generate_target(self):
        for i in range(self.max_steps):
            intervention_id = np.random.randint(0, self.num_objects)
            to_color = np.random.randint(0, self.num_colors)
            while to_color == torch.argmax(self.object_to_color[intervention_id]):
                to_color = np.random.randint(0, self.num_colors)

            self.object_to_color_target[intervention_id][to_color] = 1
            self.sample_variables_target(intervention_id)

    def reset(self, graph = None):
        self.cur_step = 0
        

        # Generate masks so that each variable only recieves input from its parents.
        self.generate_masks()

        self.object_to_color = [torch.zeros(self.num_colors) for _ in range(self.num_objects)]
        self.object_to_color_target = [torch.zeros(self.num_colors) for _ in range(self.num_objects)]
        # Sample color for root node randomly
        root_color = np.random.randint(0, self.num_colors)
        self.object_to_color[0][root_color] = 1

        # Sample color for other nodes using MLPs
        self.sample_variables(0)

        self.generate_target()

        
        return self.render()

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

    def sample_variables(self, idx):
        """
        idx: variable at which intervention is performed
        """
        
        for v in range(idx + 1, self.num_objects):
            inp = torch.cat(self.object_to_color, dim = 0).unsqueeze(0)
            mask = self.mask[v].unsqueeze(0)

            out = self.mlps[v](inp, mask)
            self.object_to_color[v] = out.squeeze(0)

    def sample_variables_target(self, idx):
        """
        idx: variable at which intervention is performed
        """
        
        for v in range(idx + 1, self.num_objects):
            inp = torch.cat(self.object_to_color_target, dim = 0).unsqueeze(0)
            mask = self.mask[v].unsqueeze(0)

            out = self.mlps[v](inp, mask)
            self.object_to_color_target[v] = out.squeeze(0)

    def translate(self, obj_id, color_id):
        """Translate object pixel.

        Args:
            obj_id: ID of object.
            color_id: ID of color.
        """
        color_ = torch.zeros(self.num_colors)
        color_[color_id] = 1
        self.object_to_color[obj_id] = color_
        self.sample_variables(obj_id)
        for idx, obj in self.objects.items():
            obj.color = torch.argmax(self.object_to_color[idx])

    def step(self, action: int):
        
        obj_id = action // self.num_colors
        color_id = action % self.num_colors 

        
        done = False
        if self.cur_step > self.max_steps:
            done = True
        self.translate(obj_id, color_id)
        matches = 0
        for c1, c2 in zip(self.object_to_color, self.object_to_color_target):
            if torch.argmax(c1) == torch.argmax(c2):
                matches+=1
        reward = matches / self.num_objects
        
        state_obs = self.render()
        return state_obs, reward, done, None