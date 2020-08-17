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
import skimage.draw

from cswm import utils
import random



mpl.use('Agg')

def random_dag(M, N, g = None):
    """Generate a random Directed Acyclic Graph (DAG) with a given number of nodes and edges."""
    if M == 3:
        return np.array([[0, 0, 0],[1, 0, 0], [1, 0, 0]])
    if M == 5:
        return np.array([[0., 0., 0., 0., 0.],
        [1., 0., 0., 0., 0.],
        [1., 0., 0., 0., 0.],
        [1., 1., 0., 0., 0.],
        [0., 0., 1., 0., 0.]]) 
    if g is None:
        expParents = 5
        idx        = np.arange(M).astype(np.float32)[:,np.newaxis]
        idx_maxed  = np.minimum(idx * 0.5, expParents)
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


class MLP(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.layers = []
        for i in range(1, len(dims)):
            self.layers.append(nn.Linear(dims[i-1], dims[i]))
            torch.nn.init.orthogonal_(self.layers[-1].weight.data, 2.5)
            torch.nn.init.uniform_(self.layers[-1].bias.data, -1.1, +1.1)
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x, mask):

        x = x * mask

        for i, l in enumerate(self.layers):
            if i == len(self.layers) - 1:
                x = torch.softmax(l(x), dim = 1)
            else:
                x = torch.relu(l(x))
        #print(x)
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
        #np.random.seed(0)
        #torch.manual_seed(0)
        self.width = width
        self.height = height
        self.render_type = render_type

        self.num_objects = num_objects
        
        if num_colors is None:
            num_colors = num_objects
        self.num_colors = num_colors
        self.num_actions = self.num_objects * self.num_colors
        self.num_target_interventions = max_steps
        self.max_steps = max_steps
        
        self.mlps = []
        self.mask = None

        colors = ['blue', 'green', 'yellow', 'white', 'red']

        self.colors, _ = utils.get_colors_and_weights(cmap = 'Set1', num_colors = self.num_colors)#[mpl.colors.to_rgba(colors[i]) for i in range(self.num_colors)]
        self.object_to_color = [torch.zeros(self.num_colors) for _ in range(self.num_objects)]

        self.np_random = None
        self.game = None
        self.target = None

        # Initialize to pos outside of env for easier collision resolution.
        self.objects = OrderedDict()

        self.adjacency_matrix = None

        mlp_dims = [self.num_objects * self.num_colors, 4 * self.num_objects * self.num_colors, self.num_colors]

        self.mlps = []

        for i in range(self.num_objects):
            self.mlps.append(MLP(mlp_dims))

        num_nodes = self.num_objects
        num_edges = np.random.randint(num_nodes, (((num_nodes) * (num_nodes - 1)) // 2) + 1)
        #if graph is None:
        self.adjacency_matrix = random_dag(num_nodes, num_edges)
        #else:
        #    self.adjacency_matrix = random_dag(num_nodes, num_nodes, g = graph)

        self.adjacency_matrix = torch.from_numpy(self.adjacency_matrix).float()

        # Generate masks so that each variable only recieves input from its parents.
        self.generate_masks()

        # If True, then check for collisions and don't allow two
        #   objects to occupy the same position.
        self.collisions = True

           


        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(3, 50, 50),
            dtype=np.float32
        )

        self.seed(seed)
        self.reset()


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def load_save_information(self, save):
        self.adjacency_matrix = save['graph']
        for i in range(self.num_objects):
            self.mlps[i].load_state_dict(save['mlp' + str(i)])
        self.generate_masks()
        self.reset()

    def set_graph(self, g):
        num_nodes = self.num_objects
        num_edges = np.random.randint(num_nodes, (((num_nodes) * (num_nodes - 1)) // 2) + 1)
        self.adjacency_matrix = random_dag(num_nodes, num_edges, g = g)
        self.adjacency_matrix = torch.from_numpy(self.adjacency_matrix).float()
        self.generate_masks()
        self.reset()

    def get_save_information(self):
        save = {}
        save['graph'] = self.adjacency_matrix
        for i in range(self.num_objects):
            save['mlp' + str(i)] = self.mlps[i].state_dict()
        return save

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
            if idx == 0:
                rr, cc = skimage.draw.circle(
                    obj.pos.x * 10 + 5, obj.pos.y * 10 + 5, 5, im.shape)
                im[rr, cc, :] = self.colors[obj.color][:3]
            elif idx == 1:
                rr, cc = triangle(
                    obj.pos.x * 10, obj.pos.y * 10, 10, im.shape)
                im[rr, cc, :] = self.colors[obj.color][:3]
            elif idx == 2:
                rr, cc = square(
                    obj.pos.x * 10, obj.pos.y * 10, 10, im.shape)
                im[rr, cc, :] = self.colors[obj.color][:3]
            elif idx == 3:
                rr, cc = diamond(
                    obj.pos.x * 10, obj.pos.y * 10, 10, im.shape)
                im[rr, cc, :] = self.colors[obj.color][:3]
            elif idx == 4:
                rr, cc = pentagon(
                    obj.pos.x * 10, obj.pos.y * 10, 10, im.shape)
                im[rr, cc, :] = self.colors[obj.color][:3]
            elif idx == 5:
                rr, cc = cross(
                    obj.pos.x * 10, obj.pos.y * 10, 10, im.shape)
                im[rr, cc, :] = self.colors[obj.color][:3]
            elif idx == 6:
                rr, cc = parallelogram(
                    obj.pos.x * 10, obj.pos.y * 10, 10, im.shape)
                im[rr, cc, :] = self.colors[obj.color][:3]
            else:

                rr, cc = scalene_triangle(
                    obj.pos.x * 10, obj.pos.y * 10, 10, im.shape)
                im[rr, cc, :] = self.colors[obj.color][:3]



        return im.transpose([2, 0, 1])


    def render_grid_target(self):
        im = np.zeros((3, self.width, self.height))
        for idx, obj in self.objects.items():
            im[:, obj.pos.x, obj.pos.y] = self.colors[torch.argmax(self.object_to_color_target[idx]).item()][:3]
        return im

    def render_circles_target(self):
        im = np.zeros((self.width * 10, self.height * 10, 3), dtype=np.float32)
        for idx, obj in self.objects.items():
            rr, cc = skimage.draw.circle(
                obj.pos.x * 10 + 5, obj.pos.y * 10 + 5, 5, im.shape)
            im[rr, cc, :] = self.colors[torch.argmax(self.object_to_color_target[idx]).item()][:3]
        return im.transpose([2, 0, 1])

    def render_shapes_target(self):
        im = np.zeros((self.width * 10, self.height * 10, 3), dtype=np.float32)
        for idx, obj in self.objects.items():
            if idx == 0:
                rr, cc = skimage.draw.circle(
                    obj.pos.x * 10 + 5, obj.pos.y * 10 + 5, 5, im.shape)
                im[rr, cc, :] = self.colors[torch.argmax(self.object_to_color_target[idx]).item()][:3]
            elif idx == 1:
                rr, cc = triangle(
                    obj.pos.x * 10, obj.pos.y * 10, 10, im.shape)
                im[rr, cc, :] = self.colors[torch.argmax(self.object_to_color_target[idx]).item()][:3]
            elif idx == 2:
                rr, cc = square(
                    obj.pos.x * 10, obj.pos.y * 10, 10, im.shape)
                im[rr, cc, :] = self.colors[torch.argmax(self.object_to_color_target[idx]).item()][:3]
            elif idx == 3:
                rr, cc = diamond(
                    obj.pos.x * 10, obj.pos.y * 10, 10, im.shape)
                im[rr, cc, :] = self.colors[torch.argmax(self.object_to_color_target[idx]).item()][:3]
            elif idx == 4:
                rr, cc = pentagon(
                    obj.pos.x * 10, obj.pos.y * 10, 10, im.shape)
                im[rr, cc, :] = self.colors[torch.argmax(self.object_to_color_target[idx]).item()][:3]
            elif idx == 5:
                rr, cc = cross(
                    obj.pos.x * 10, obj.pos.y * 10, 10, im.shape)
                im[rr, cc, :] = self.colors[torch.argmax(self.object_to_color_target[idx]).item()][:3]
            elif idx == 6:
                rr, cc = parallelogram(
                    obj.pos.x * 10, obj.pos.y * 10, 10, im.shape)
                im[rr, cc, :] = self.colors[torch.argmax(self.object_to_color_target[idx]).item()][:3]
            else:

                rr, cc = scalene_triangle(
                    obj.pos.x * 10, obj.pos.y * 10, 10, im.shape)
                im[rr, cc, :] = self.colors[torch.argmax(self.object_to_color_target[idx]).item()][:3]



        return im.transpose([2, 0, 1])

    def render_cubes(self):
        im = render_cubes(self.objects, self.width)
        return im.transpose([2, 0, 1])

    def render(self):
        return np.concatenate((dict(
            grid=self.render_grid,
            circles=self.render_circles,
            shapes=self.render_shapes,
            cubes=self.render_cubes,
        )[self.render_type](), dict(
            grid=self.render_grid_target,
            circles=self.render_circles_target,
            shapes=self.render_shapes_target,
            cubes=self.render_cubes,
        )[self.render_type]()), axis = 0) 

    def get_state(self):
        im = np.zeros(
            (self.num_objects * self.num_colors, self.width, self.height), dtype=np.int32)
        im_target = np.zeros(
            (self.num_objects * self.num_colors, self.width, self.height), dtype=np.int32)
        for idx, obj in self.objects.items():
            im[idx * self.num_colors + obj.color, obj.pos.x, obj.pos.y] =  1
            im_target[idx * self.num_colors + torch.argmax(self.object_to_color_target[idx]).item(), obj.pos.x, obj.pos.y] =  1
        return im, im_target

    def generate_masks(self):
        mask = self.adjacency_matrix.unsqueeze(-1)
        mask = mask.repeat(1, 1, self.num_colors)
        self.mask = mask.view(self.adjacency_matrix.size(0), -1)

    def generate_target(self):
        for i in range(self.num_target_interventions):
            intervention_id = random.randint(0, self.num_objects - 1)
            to_color = random.randint(0, self.num_colors - 1)
            #while to_color == torch.argmax(self.object_to_color[intervention_id]):
            #    to_color = random.randint(0, self.num_colors - 1)

            self.object_to_color_target[intervention_id][to_color] = 1
            self.sample_variables_target(intervention_id)

    def check_softmax(self):
        s_ = []
        for i in range(1, len(self.objects)):
            x = torch.cat(self.object_to_color, dim = 0).unsqueeze(0)
            mask = self.mask[i].unsqueeze(0)
            _, s = self.mlps[i](x, mask, return_softmax = True)
            s_.append(s.detach().cpu().numpy().tolist())
        return s_
        

    def check_softmax_target(self):
        s_ = []
        for i in range(1, len(self.objects)):
            x = torch.cat(self.object_to_color_target, dim = 0).unsqueeze(0)
            mask = self.mask[i].unsqueeze(0)
            _, s = self.mlps[i](x, mask, return_softmax = True)
            s_.append(s.detach().cpu().numpy().tolist())
        return s_

    def reset(self, graph = None):
        self.cur_step = 0

        self.object_to_color = [torch.zeros(self.num_colors) for _ in range(self.num_objects)]
        self.object_to_color_target = [torch.zeros(self.num_colors) for _ in range(self.num_objects)]

        # Sample color for root node randomly
        root_color = np.random.randint(0, self.num_colors)
        self.object_to_color[0][root_color] = 1
        self.object_to_color_target[0][root_color] = 1


         # Sample color for other nodes using MLPs
        self.sample_variables(0, do_everything = True)

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
        self.sample_variables_target(0, do_everything = True)

        self.generate_target()
        #self.check_softmax()
        #self.check_softmax_target()
        observations = self.render()
        observation_in, observations_target = observations[:3, :, :], observations[3:, :, :]
        state_in, state_target = self.get_state()
        return (state_in, observation_in), (observations_target, state_target)

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

    def is_reachable(self, idx, reached):
        for r in reached:
            if self.adjacency_matrix[idx, r] == 1:
                return True
        return False


    def sample_variables(self, idx, do_everything = False):
        """
        idx: variable at which intervention is performed
        """
        reached = [idx]
        for v in range(idx + 1, self.num_objects):
            if do_everything or self.is_reachable(v, reached):
                    reached.append(v)
                    inp = torch.cat(self.object_to_color, dim = 0).unsqueeze(0)
                    mask = self.mask[v].unsqueeze(0)

                    out = self.mlps[v](inp, mask)
                    self.object_to_color[v] = out.squeeze(0)

    def sample_variables_target(self, idx, do_everything = False):
        """
        idx: variable at which intervention is performed
        """
        reached = [idx]
        for v in range(idx + 1, self.num_objects):
            if do_everything or self.is_reachable(v, reached):
                reached.append(v)

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
        
        self.translate(obj_id, color_id)
        matches = 0
        for c1, c2 in zip(self.object_to_color, self.object_to_color_target):
            if torch.argmax(c1).item() == torch.argmax(c2).item():
                matches+=1
        reward = 0
        #reward = 0
        #if matches == self.num_objects:
        #    reward = 1

        
        state_obs = self.render()
        state_obs = state_obs[:3, :, :]
        state = self.get_state()[0]
        state_obs = (state, state_obs)
        if self.cur_step >= self.max_steps:
            done = True
        reward = matches / self.num_objects
        self.cur_step += 1
        return state_obs, reward, done, None
