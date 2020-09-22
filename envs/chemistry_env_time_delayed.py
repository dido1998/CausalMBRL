"""Gym environment for chnging colors of shapes."""

import numpy as np
import torch
import torch.nn as nn
import re

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
import math
import copy

graphs = {
    'chain3':'0->1->2',
    'fork3':'0->{1-2}',
    'collider3':'{0-1}->2',
    'collider4':'{0-2}->3',
    'collider5':'{0-3}->4',
    'collider6':'{0-4}->5',
    'collider7':'{0-5}->6',
    'collider8':'{0-6}->7',
    'collider9':'{0-7}->8',
    'collider10':'{0-8}->9',
    'collider11':'{0-9}->10',
    'collider12':'{0-10}->11',
    'collider13':'{0-11}->12',
    'collider14':'{0-12}->13',
    'collider15':'{0-13}->14',
    'confounder3':'{0-2}->{0-2}',
    'chain4':'0->1->2->3',
    'chain5':'0->1->2->3->4',
    'chain6':'0->1->2->3->4->5',
    'chain7':'0->1->2->3->4->5->6',
    'chain8':'0->1->2->3->4->5->6->7',
    'chain9':'0->1->2->3->4->5->6->7->8',
    'chain10':'0->1->2->3->4->5->6->7->8->9',
    'chain11':'0->1->2->3->4->5->6->7->8->9->10',
    'chain12':'0->1->2->3->4->5->6->7->8->9->10->11',
    'chain13':'0->1->2->3->4->5->6->7->8->9->10->11->12',
    'chain14':'0->1->2->3->4->5->6->7->8->9->10->11->12->13',
    'chain15':'0->1->2->3->4->5->6->7->8->9->10->11->12->13->14',
    'full3':'{0-2}->{0-2}',
    'full4':'{0-3}->{0-3}',
    'full5':'{0-4}->{0-4}',
    'full6':'{0-5}->{0-5}',
    'full7':'{0-6}->{0-6}',
    'full8':'{0-7}->{0-7}',
    'full9':'{0-8}->{0-8}',
    'full10':'{0-9}->{0-9}',
    'full11':'{0-10}->{0-10}',
    'full12':'{0-11}->{0-11}',
    'full13':'{0-12}->{0-12}',
    'full14':'{0-13}->{0-13}',
    'full15':'{0-14}->{0-14}',
    'tree9':'0->1->3->7,0->2->6,1->4,3->8,2->5',
    'tree10':'0->1->3->7,0->2->6,1->4->9,3->8,2->5',
    'tree11':'0->1->3->7,0->2->6,1->4->10,3->8,4->9,2->5',
    'tree12':'0->1->3->7,0->2->6,1->4->10,3->8,4->9,2->5->11',
    'tree13':'0->1->3->7,0->2->6,1->4->10,3->8,4->9,2->5->11,5->12',
    'tree14':'0->1->3->7,0->2->6,1->4->10,3->8,4->9,2->5->11,5->12,6->13',
    'tree15':'0->1->3->7,0->2->6->14,1->4->10,3->8,4->9,2->5->11,5->12,6->13',
    'jungle3':'0->{1-2}',
    'jungle4':'0->1->3,0->2,0->3',
    'jungle5':'0->1->3,1->4,0->2,0->3,0->4',
    'jungle6':'0->1->3,1->4,0->2->5,0->3,0->4,0->5',
    'jungle7':'0->1->3,1->4,0->2->5,2->6,0->3,0->4,0->5,0->6',
    'jungle8':'0->1->3->7,1->4,0->2->5,2->6,0->3,0->4,0->5,0->6,1->7',
    'jungle9':'0->1->3->7,3->8,1->4,0->2->5,2->6,0->3,0->4,0->5,0->6,1->7,1->8',
    'jungle10':'0->1->3->7,3->8,1->4->9,0->2->5,2->6,0->3,0->4,0->5,0->6,1->7,1->8,1->9',
    'jungle11':'0->1->3->7,3->8,1->4->9,4->10,0->2->5,2->6,0->3,0->4,0->5,0->6,1->7,1->8,1->9,1->10',
    'jungle12':'0->1->3->7,3->8,1->4->9,4->10,0->2->5->11,2->6,0->3,0->4,0->5,0->6,1->7,1->8,1->9,1->10,2->11',
    'jungle13':'0->1->3->7,3->8,1->4->9,4->10,0->2->5->11,5->12,2->6,0->3,0->4,0->5,0->6,1->7,1->8,1->9,1->10,2->11,2->12',
    'jungle14':'0->1->3->7,3->8,1->4->9,4->10,0->2->5->11,5->12,2->6->13,0->3,0->4,0->5,0->6,1->7,1->8,1->9,1->10,2->11,2->12,2->13',
    'jungle15':'0->1->3->7,3->8,1->4->9,4->10,0->2->5->11,5->12,2->6->13,6->14,0->3,0->4,0->5,0->6,1->7,1->8,1->9,1->10,2->11,2->12,2->13,2->14',
    'bidiag3':'{0-2}->{0-2}',
    'bidiag4':'{0-1}->{1-2}->{2-3}',
    'bidiag5':'{0-1}->{1-2}->{2-3}->{3-4}',
    'bidiag6':'{0-1}->{1-2}->{2-3}->{3-4}->{4-5}',
    'bidiag7':'{0-1}->{1-2}->{2-3}->{3-4}->{4-5}->{5-6}',
    'bidiag8':'{0-1}->{1-2}->{2-3}->{3-4}->{4-5}->{5-6}->{6-7}',
    'bidiag9':'{0-1}->{1-2}->{2-3}->{3-4}->{4-5}->{5-6}->{6-7}->{7-8}',
    'bidiag10':'{0-1}->{1-2}->{2-3}->{3-4}->{4-5}->{5-6}->{6-7}->{7-8}->{8-9}',
    'bidiag11':'{0-1}->{1-2}->{2-3}->{3-4}->{4-5}->{5-6}->{6-7}->{7-8}->{8-9}->{9-10}',
    'bidiag12':'{0-1}->{1-2}->{2-3}->{3-4}->{4-5}->{5-6}->{6-7}->{7-8}->{8-9}->{9-10}->{10-11}',
    'bidiag13':'{0-1}->{1-2}->{2-3}->{3-4}->{4-5}->{5-6}->{6-7}->{7-8}->{8-9}->{9-10}->{10-11}->{11-12}',
    'bidiag14':'{0-1}->{1-2}->{2-3}->{3-4}->{4-5}->{5-6}->{6-7}->{7-8}->{8-9}->{9-10}->{10-11}->{11-12}->{12-13}',
    'bidiag15':'{0-1}->{1-2}->{2-3}->{3-4}->{4-5}->{5-6}->{6-7}->{7-8}->{8-9}->{9-10}->{10-11}->{11-12}->{12-13}->{13-14}',
}

def parse_skeleton(graph, M=None):
    """
    Parse the skeleton of a causal graph in the mini-language of --graph.
    
    The mini-language is:
        
        GRAPH      = ""
                     CHAIN{, CHAIN}*
        CHAIN      = INT_OR_SET {-> INT_OR_SET}
        INT_OR_SET = INT | SET
        INT        = [0-9]*
        SET        = \{ SET_ELEM {, SET_ELEM}* \}
        SET_ELEM   = INT | INT_RANGE
        INT_RANGE  = INT - INT
    """
    
    regex = re.compile(r'''
        \s*                                      # Skip preceding whitespace
        (                                        # The set of tokens we may capture, including
          [,]                                  | # Commas
          (?:\d+)                              | # Integers
          (?:                                    # Integer set:
            \{                                   #   Opening brace...
              \s*                                #   Whitespace...
              \d+\s*(?:-\s*\d+\s*)?              #   First integer (range) in set...
              (?:,\s*\d+\s*(?:-\s*\d+\s*)?\s*)*  #   Subsequent integers (ranges)
            \}                                   #   Closing brace...
          )                                    | # End of integer set.
          (?:->)                                 # Arrows
        )
    ''', re.A|re.X)
    
    # Utilities
    def parse_int(s):
        try:    return int(s.strip())
        except: return None
    
    def parse_intrange(s):
        try:
            sa, sb = map(str.strip, s.strip().split("-", 1))
            sa, sb = int(sa), int(sb)
            sa, sb = min(sa,sb), max(sa,sb)+1
            return range(sa,sb)
        except:
            return None
    
    def parse_intset(s):
        try:
            i = set()
            for s in map(str.strip, s.strip()[1:-1].split(",")):
                if parse_int(s) is not None: i.add(parse_int(s))
                else:                        i.update(set(parse_intrange(s)))
            return sorted(i)
        except:
            return None
    
    def parse_either(s):
        asint = parse_int(s)
        if asint is not None: return asint
        asset = parse_intset(s)
        if asset is not None: return asset
        raise ValueError
    
    def find_max(chains):
        m = 0
        for chain in chains:
            for link in chain:
                link = max(link) if isinstance(link, list) else link
                m = max(link, m)
        return m
    
    # Crack the string into a list of lists of (ints | lists of ints)
    graph  = [graph] if isinstance(graph, str) else graph
    chains = []
    for gstr in graph:
        for chain in re.findall("((?:[^,{]+|\{.*?\})+)+", gstr, re.A):
            links = list(map(str.strip, regex.findall(chain)))
            assert(len(links)&1)
            
            chain = [parse_either(links.pop(0))]
            while links:
                assert links.pop(0) == "->"
                chain.append(parse_either(links.pop(0)))
            chains.append(chain)
    
    # Find the maximum integer referenced within the skeleton
    uM = find_max(chains)+1
    if M is None:
        M = uM
    else:
        assert(M >= uM)
        M = max(M, uM)
    
    # Allocate adjacency matrix.
    gamma = np.zeros((M,M), dtype=np.float32)
    
    # Interpret the skeleton
    for chain in chains:
        for prevlink, nextlink in zip(chain[:-1], chain[1:]):
            if   isinstance(prevlink, list) and isinstance(nextlink, list):
                for i in nextlink:
                    for j in prevlink:
                        if i>j:
                            gamma[i,j] = 1
            elif isinstance(prevlink, list) and isinstance(nextlink, int):
                for j in prevlink:
                    if nextlink>j:
                        gamma[nextlink,j] = 1
            elif isinstance(prevlink, int)  and isinstance(nextlink, list):
                minn = min(nextlink)
                if   minn == prevlink:
                    raise ValueError("Edges are not allowed from " +
                                     str(prevlink) + " to oneself!")
                elif minn <  prevlink:
                    raise ValueError("Edges are not allowed from " +
                                     str(prevlink) + " to ancestor " +
                                     str(minn) + " !")
                else:
                    for i in nextlink:
                        gamma[i,prevlink] = 1
            elif isinstance(prevlink, int)  and isinstance(nextlink, int):
                if   nextlink == prevlink:
                    raise ValueError("Edges are not allowed from " +
                                     str(prevlink) + " to oneself!")
                elif nextlink <  prevlink:
                    raise ValueError("Edges are not allowed from " +
                                     str(prevlink) + " to ancestor " +
                                     str(nextlink) + " !")
                else:
                    gamma[nextlink,prevlink] = 1
    
    # Return adjacency matrix.
    return gamma


mpl.use('Agg')

def linear_gradient(start_hex, finish_hex, n=5):
  ''' returns a gradient list of (n) colors between
    two hex colors. start_hex and finish_hex
    should be the full six-digit color string,
    inlcuding the number sign ("#FFFFFF") '''
  # Starting and ending colors in RGB form
  s = start_hex
  f = finish_hex


  # Initilize a list of the output colors with the starting color
  RGB_list = [s]
  # Calcuate a color at each evenly spaced value of t from 1 to n
  for t in range(1, n):
    # Interpolate RGB vector for color at the current value of t
    curr_vector = [
      int(s[j] + (float(t)/(n-1))*(f[j]-s[j]))
      for j in range(3)
    ]
    # Add it to our list of output colors
    RGB_list.append(curr_vector)
    
  return RGB_list

def random_dag(M, N, g = None):
    """Generate a random Directed Acyclic Graph (DAG) with a given number of nodes and edges."""
    if g is None:
        expParents = 5
        idx        = np.arange(M).astype(np.float32)[:,np.newaxis]
        idx_maxed  = np.minimum(idx * 0.5, expParents)
        p          = np.broadcast_to(idx_maxed/(idx+1), (M, M))
        B          = np.random.binomial(1, p)
        B          = np.tril(B, -1)
        return B
    else:
        gammagt = parse_skeleton(g, M=M)
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
            torch.nn.init.orthogonal_(self.layers[-1].weight.data, 3.5)
            torch.nn.init.uniform_(self.layers[-1].bias.data, -2.1, +2.1)
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
class Color:
    r: float
    g: float
    b: float


@dataclass
class Object:
    pos: Coord
    color: Color 



class ColorChangingTimeRL(gym.Env):
    """Gym environment for block pushing task."""

    def __init__(self, width=5, height=5, render_type='cubes',
                 *, num_objects=5,
                 num_colors=None,  movement = 'Dynamic', edge = -1, max_steps = 50, seed=None):
        #np.random.seed(0)
        #torch.manual_seed(0)
        self.width = width
        self.height = height
        self.render_type = render_type

        self.num_objects = num_objects
        self.movement = movement
        
        if num_colors is None:
            num_colors = num_objects
        self.num_colors = num_colors
        self.num_actions = self.num_objects * self.num_colors
        self.num_target_interventions = max_steps
        self.max_steps = max_steps
        self.edge = edge
        
        self.mlps = []
        self.mask = None

        colors = ['blue', 'green', 'yellow', 'gray', 'red']
        self.colors = []
        for c in colors:
            rgb = mpl.colors.to_rgb(c)
            rgb = [int(r * 255) for r in rgb]
            self.colors.append(rgb)

        #self.colors, _ = utils.get_colors_and_weights(cmap = 'Set1', num_colors = self.num_colors)#[mpl.colors.to_rgba(colors[i]) for i in range(self.num_colors)]
        self.color_tracker = [[self.colors[random.randint(0, self.num_colors - 1)]] for c in range(self.num_objects)]

        self.np_random = None
        self.game = None
        self.target = None

        # Initialize to pos outside of env for easier collision resolution.
        self.objects = OrderedDict()

        self.adjacency_matrix = None

        mlp_dims = [self.num_objects * 3, 4 * self.num_objects, self.num_colors]

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
        if self.edge == -1:
            self.times = torch.ones(self.adjacency_matrix.size()) * 3

            self.times = (torch.poisson(self.times) + 1) * self.adjacency_matrix
        else:
            self.times = torch.ones(self.adjacency_matrix.size()) * self.edge * self.adjacency_matrix
        
        #print(self.adjacency_matrix)

        # Generate masks so that each variable only recieves input from its parents.
        self.generate_masks()

        # If True, then check for collisions and don't allow two
        #   objects to occupy the same position.
        self.collisions = True

        self.objects = OrderedDict()
        # Randomize object position.
        fixed_object_to_position_mapping = [(0, 0), (0, 4), (4, 0), (4, 4), (2, 2), (1,1), (1, 3), (3, 1)]
        while len(self.objects) < self.num_objects:
            idx = len(self.objects)
            # Re-sample to ensure objects don't fall on same spot.
            while not (idx in self.objects and
                       self.valid_pos(self.objects[idx].pos, idx)):
                color = self.color_tracker[idx][-1]
                self.objects[idx] = Object(
                    pos=Coord(
                        x=fixed_object_to_position_mapping[idx][0],
                        y=fixed_object_to_position_mapping[idx][1],
                    ),
                    color=Color(r = color[0], g = color[1], b = color[2]))   


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
        self.times = save['times']
        for i in range(self.num_objects):
            self.mlps[i].load_state_dict(save['mlp' + str(i)])
        self.generate_masks()
        self.reset()

    def set_graph(self, g):
        if g in graphs.keys():
            if int(g[-1]) != self.num_objects:
                print('ERROR:Env created for ' + str(self.num_objects) + ' objects while graph specified for ' + g[-1] + ' objects')
                exit()
            print('INFO: Loading predefined graph for configuration '+str(g))
            g = graphs[g]
        num_nodes = self.num_objects
        num_edges = np.random.randint(num_nodes, (((num_nodes) * (num_nodes - 1)) // 2) + 1)
        self.adjacency_matrix = random_dag(num_nodes, num_edges, g = g)
        self.adjacency_matrix = torch.from_numpy(self.adjacency_matrix).float()
        if self.edge == -1:
            self.times = torch.ones(self.adjacency_matrix.size()) * 3
            self.times = (torch.poisson(self.times) + 1) * self.adjacency_matrix
        else:
            self.times = torch.ones(self.adjacency_matrix.size()) * self.edge * self.adjacency_matrix
        print(self.adjacency_matrix)
        print(self.times)
        self.generate_masks()
        self.reset()

    def get_save_information(self):
        save = {}
        save['graph'] = self.adjacency_matrix
        save['times'] = self.times
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
            color = np.array([obj.color.r, obj.color.g, obj.color.b])
            if idx == 0:
                rr, cc = skimage.draw.circle(
                    obj.pos.x * 10 + 5, obj.pos.y * 10 + 5, 5, im.shape)
                im[rr, cc, :] = color
            elif idx == 1:
                rr, cc = triangle(
                    obj.pos.x * 10, obj.pos.y * 10, 10, im.shape)
                im[rr, cc, :] = color
            elif idx == 2:
                rr, cc = square(
                    obj.pos.x * 10, obj.pos.y * 10, 10, im.shape)
                im[rr, cc, :] = color
            elif idx == 3:
                rr, cc = diamond(
                    obj.pos.x * 10, obj.pos.y * 10, 10, im.shape)
                im[rr, cc, :] = color
            elif idx == 4:
                rr, cc = pentagon(
                    obj.pos.x * 10, obj.pos.y * 10, 10, im.shape)
                im[rr, cc, :] = color
            elif idx == 5:
                rr, cc = cross(
                    obj.pos.x * 10, obj.pos.y * 10, 10, im.shape)
                im[rr, cc, :] = color
            elif idx == 6:
                rr, cc = parallelogram(
                    obj.pos.x * 10, obj.pos.y * 10, 10, im.shape)
                im[rr, cc, :] = color
            else:

                rr, cc = scalene_triangle(
                    obj.pos.x * 10, obj.pos.y * 10, 10, im.shape)
                im[rr, cc, :] = color



        return im.transpose([2, 0, 1])


    def render_grid_target(self):
        im = np.zeros((3, self.width, self.height))
        for idx, obj in self.objects_target.items():
            im[:, obj.pos.x, obj.pos.y] = self.colors[torch.argmax(self.object_to_color_target[idx]).item()][:3]
        return im

    def render_circles_target(self):
        im = np.zeros((self.width * 10, self.height * 10, 3), dtype=np.float32)
        for idx, obj in self.objects_target.items():
            color = np.array([obj.color.r, obj.color.g, obj.color.b])
            rr, cc = skimage.draw.circle(
                obj.pos.x * 10 + 5, obj.pos.y * 10 + 5, 5, im.shape)
            im[rr, cc, :] = color
        return im.transpose([2, 0, 1])

    def render_shapes_target(self):
        im = np.zeros((self.width * 10, self.height * 10, 3), dtype=np.float32)
        for idx, obj in self.objects_target.items():
            color = np.array([obj.color.r, obj.color.g, obj.color.b])
            if idx == 0:
                rr, cc = skimage.draw.circle(
                    obj.pos.x * 10 + 5, obj.pos.y * 10 + 5, 5, im.shape)
                im[rr, cc, :] = color
            elif idx == 1:
                rr, cc = triangle(
                    obj.pos.x * 10, obj.pos.y * 10, 10, im.shape)
                im[rr, cc, :] = color
            elif idx == 2:
                rr, cc = square(
                    obj.pos.x * 10, obj.pos.y * 10, 10, im.shape)
                im[rr, cc, :] = color
            elif idx == 3:
                rr, cc = diamond(
                    obj.pos.x * 10, obj.pos.y * 10, 10, im.shape)
                im[rr, cc, :] = color
            elif idx == 4:
                rr, cc = pentagon(
                    obj.pos.x * 10, obj.pos.y * 10, 10, im.shape)
                im[rr, cc, :] = color
            elif idx == 5:
                rr, cc = cross(
                    obj.pos.x * 10, obj.pos.y * 10, 10, im.shape)
                im[rr, cc, :] = color
            elif idx == 6:
                rr, cc = parallelogram(
                    obj.pos.x * 10, obj.pos.y * 10, 10, im.shape)
                im[rr, cc, :] = color
            else:
                rr, cc = scalene_triangle(
                    obj.pos.x * 10, obj.pos.y * 10, 10, im.shape)
                im[rr, cc, :] = color



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
        )[self.render_type]()), axis = 0)  / 255.0

    def get_state(self):
        im = np.zeros(
            (3, self.width, self.height), dtype=np.int32)
        im_target = np.zeros(
            (3, self.width, self.height), dtype=np.int32)
        for idx, obj in self.objects.items():
            im[:, obj.pos.x, obj.pos.y] =  [obj.color.r, obj.color.g, obj.color.b]
        for idx, obj in self.objects_target.items():
            im[:, obj.pos.x, obj.pos.y] =  [obj.color.r, obj.color.g, obj.color.b]
        return im, im_target

    def generate_masks(self):
        mask = self.adjacency_matrix.unsqueeze(-1)
        mask = mask.repeat(1, 1, 3)
        self.mask = mask.view(self.adjacency_matrix.size(0), -1)

    def generate_target(self, num_steps = 10):
        for i in range(num_steps):
            intervention_id = random.randint(0, self.num_objects - 1)
            to_color = self.colors[random.randint(0, self.num_colors - 1)]
            #while to_color == torch.argmax(self.object_to_color[intervention_id]):
            #    to_color = random.randint(0, self.num_colors - 1)
            self.objects_target[intervention_id].color.r = to_color[0]
            self.objects_target[intervention_id].color.g = to_color[1]
            self.objects_target[intervention_id].color.b = to_color[2]

            self.sample_variables(intervention_id, targets = True)
            self.step_variables(targets = True)
        

    def reset(self, num_steps = 10, graph = None):
        self.cur_step = 0        
        #print('in reset')
        # Sample
        self.sample_variables(0, do_everything = True, targets = False)
        self.step_variables(targets = False)

        
        
        if self.movement == 'Dynamic':
            temp_object = self.objects.copy()
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
                        color=temp_object[idx].color)

        self.objects_target = copy.deepcopy(self.objects)
        target_color = self.colors[random.randint(0, self.num_colors - 1)]
        self.objects_target[0].color.r = target_color[0]
        self.objects_target[0].color.g = target_color[1]
        self.objects_target[0].color.b = target_color[2]
        
        self.sample_variables(0, do_everything = True, targets = True)
        self.step_variables(targets = True)

        self.generate_target(num_steps)
        #self.check_softmax()
        #self.check_softmax_target()


        observations = self.render()
        observation_in, observations_target = observations[:3, :, :], observations[3:, :, :]
        state_in, state_target = self.get_state()
        return (state_in, observation_in), (state_target, observations_target)

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
                return r, True
        return None, False


    def sample_variables(self, idx, do_everything = False, targets = False):
        """
        idx: variable at which intervention is performed
        """
        if not targets:
            objects = copy.deepcopy(self.objects)
        else:
            objects = copy.deepcopy(self.objects_target)
        reached = [idx]

        inp = torch.zeros(1, self.num_objects * 3)
        for i, obj in objects.items():
            inp[0, i * 3] = obj.color.r
            inp[0, i * 3 + 1] = obj.color.g
            inp[0, i * 3 + 2] = obj.color.b
        for v in range(idx + 1, self.num_objects):
            parent, decision = self.is_reachable(v, reached)
            if do_everything:
                parent = idx
            if do_everything or decision:
                    #print(v, end = ' ')
                    #print(parent)

                    reached.append(v)

                    mask = self.mask[v].unsqueeze(0)

                    out = self.mlps[v](inp, mask).squeeze(0)
                    target_color = self.colors[torch.argmax(out)][:3]
                    source_color = [objects[v].color.r, objects[v].color.g, objects[v].color.b]
                    if int(self.times[v, parent].item()) > 0:

                        color_gradient = linear_gradient(source_color, target_color, n = int(self.times[v, parent].item()))
                        self.color_tracker[v] = []

                        if len(color_gradient) > 1:
                            color_gradient = color_gradient[1:]
                        color_gradient.reverse()
                        self.color_tracker[v] = color_gradient
                    else:
                        #print('here')
                        objects[v].color.r = target_color[0]
                        objects[v].color.g = target_color[1]
                        objects[v].color.b = target_color[2]
        if not targets:
            self.objects = copy.deepcopy(objects)
        else:
            self.objects_target = copy.deepcopy(objects)
    
    def step_variables(self, targets = False):
        if not targets:
            objects = copy.deepcopy(self.objects)
        else:
            objects = copy.deepcopy(self.objects_target)
        for idx, obj in objects.items():
            #print(self.color_tracker[idx])
            if len(self.color_tracker[idx]) > 0:
                next_color = self.color_tracker[idx][-1]
                self.color_tracker[idx] = self.color_tracker[idx][:-1]
                obj.color.r = next_color[0]
                obj.color.g = next_color[1]
                obj.color.b = next_color[2]
            #print(self.color_tracker[idx])
            #print('---------------')
        if not targets:
            self.objects = copy.deepcopy(objects)
        else:
            self.objects_target = copy.deepcopy(objects)

    def translate(self, obj_id, color_id):
        """Translate object pixel.

        Args:
            obj_id: ID of object.
            color_id: ID of color.
        """
        #for idx, obj in self.objects.items():
        #    print(obj.color.r, end = ' ')
        #    print(obj.color.g, end = ' ')
        #    print(obj.color.b, end = ' ')
        #    print()
        #print('----')
        color = self.colors[color_id]
        self.objects[obj_id].color.r = color[0]
        self.objects[obj_id].color.g = color[1]
        self.objects[obj_id].color.b = color[2]
        self.sample_variables(obj_id)
        #for c in self.color_tracker:
        #    print(c)
        #print('-----')
        self.step_variables()
        


    def step(self, action: int):
        
        obj_id = action // self.num_colors
        color_id = action % self.num_colors 

        
        done = False
        
        self.translate(obj_id, color_id)
        matches = 0
        for idx, obj in self.objects.items():
            color1 = [obj.color.r, obj.color.g, obj.color.b]
            color2 = [self.objects_target[idx].color.r, self.objects_target[idx].color.g, self.objects_target[idx].color.b]
            mse1 = abs(color1[0] - color2[0])
            mse2 = abs(color1[1] - color2[1])
            mse3 = abs(color1[2] - color2[2])
            mse_avg = (mse1 + mse2 + mse3) / 3
            if mse_avg < 25:
                matches += 1

        reward = 0
        #reward = 0
        #if matches == self.num_objects:
        #    reward = 1

        
        state_obs = self.render()
        state_obs = state_obs[:3, :, :]
        state = self.get_state()[0]
        state_obs = (state, state_obs)
        #if self.cur_step >= self.max_steps:
        #    done = True
        reward = matches / self.num_objects
        self.cur_step += 1
        return state_obs, reward, done, None

    def sample_step(self, action: int):
        
        obj_id = action // self.num_colors
        color_id = action % self.num_colors 

        
        done = False
        objects = self.objects.copy()
        
        self.translate(obj_id, color_id)
        matches = 0
        for idx, obj in self.objects.items():
            color1 = [obj.color.r, obj.color.g, obj.color.b]
            color2 = [self.objects_target[idx].color.r, self.objects_target[idx].color.g, self.objects_target[idx].color.b]
            mse1 = abs(color1[0] - color2[0])
            mse2 = abs(color1[1] - color2[1])
            mse3 = abs(color1[2] - color2[2])
            mse_avg = (mse1 + mse2 + mse3) / 3
            if mse_avg < 25:
                matches += 1

        reward = 0
        self.objects = objects
        
        #reward = 0
        #if matches == self.num_objects:
        #    reward = 1

        
        state_obs = self.render()
        state_obs = state_obs[:3, :, :]
        #state = self.get_state()[0]
        #state_obs = (state, state_obs)
        #if self.cur_step >= self.max_steps:
        #    done = True
        reward = matches / self.num_objects
        self.cur_step += 1
        return reward, state_obs
