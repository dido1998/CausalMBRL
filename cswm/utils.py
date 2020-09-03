"""Utility functions."""

import os
from collections import defaultdict

import h5py
import numpy as np

import torch
from torch.utils import data
from torch import nn
import torch.nn.functional as F
from torchvision.utils import save_image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from cswm.models.losses import *

EPS = 1e-17


class OneHot:
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, inputs):
        onehot = np.zeros(self.num_classes, dtype='float32')
        onehot[inputs] = 1
        return onehot


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


def save_dict_h5py(array_dict, fname):
    """Save dictionary containing numpy arrays to h5py file."""

    # Ensure directory exists
    directory = os.path.dirname(fname)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with h5py.File(fname, 'w') as hf:
        for key in array_dict.keys():
            hf.create_dataset(key, data=array_dict[key])


def load_dict_h5py(fname):
    """Restore dictionary containing numpy arrays from h5py file."""
    array_dict = dict()
    with h5py.File(fname, 'r') as hf:
        for key in hf.keys():
            array_dict[key] = hf[key][:]
    return array_dict


def save_list_dict_h5py(array_dict, fname):
    """Save list of dictionaries containing numpy arrays to h5py file."""

    # Ensure directory exists
    directory = os.path.dirname(fname)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with h5py.File(fname, 'w') as hf:
        for i in range(len(array_dict)):
            grp = hf.create_group(str(i))
            for key in array_dict[i].keys():
                grp.create_dataset(key, data=array_dict[i][key])


def load_list_dict_h5py(fname):
    """Restore list of dictionaries containing numpy arrays from h5py file."""
    array_dict = list()
    with h5py.File(fname, 'r') as hf:
        for i, grp in enumerate(hf.keys()):
            array_dict.append(dict())
            for key in hf[grp].keys():
                array_dict[i][key] = hf[grp][key][:]
    return array_dict

def get_cmap(cmap, mode):
    length = 9
    if cmap == 'Sets':
        if "FewShot" not in mode:
            cmap = plt.get_cmap('Set1')
        else:
            cmap = [plt.get_cmap('Set1'), plt.get_cmap('Set3')]
            length = [9,12]
    else :
        if "FewShot" not in mode:
            cmap = plt.get_cmap('Pastel1')
        else:
            cmap = [plt.get_cmap('Pastel1'), plt.get_cmap('Pastel2')]
            length = [9,8]

    return cmap, length

def observed_colors(num_colors, mode):
    if mode == 'ZeroShot':
        c = np.sort(np.random.uniform(0.0, 1.0, size=num_colors))
    else:
        c = (np.arange(num_colors)) / (num_colors-1)
        diff = 1.0 / (num_colors - 1)
        if mode == 'Train':
            diff = diff / 8.0
        elif mode == 'Test-v1':
            diff = diff / 4.0
        elif mode == 'Test-v2':
            diff = diff / 3.0
        elif mode == 'Test-v3':
            diff = diff / 2.0

        unif = np.random.uniform(-diff+EPS, diff-EPS, size=num_colors)
        unif[0] = abs(unif[0])
        unif[-1] = -abs(unif[-1])

        c = c + unif

    return c

def unobserved_colors(cmap, num_colors, mode, new_colors=None):
    if mode in ['Train', 'ZeroShotShape']:
        cm, length = get_cmap(cmap, mode)
        weights = np.sort(np.random.choice(length, num_colors, replace=False))
        colors = [cm(i/length) for i in weights]
    else:
        cm, length = get_cmap(cmap, mode)
        cm1, cm2 = cm
        length1, length2 = length
        l = length1 + len(new_colors)
        w = np.sort(np.random.choice(l, num_colors, replace=False))
        colors = []
        weights = []
        for i in w:
            if i < length1:
                colors.append(cm1(i/length1))
                weights.append(i)
            else:
                colors.append(cm2(new_colors[i - length1] / length2))
                weights.append(new_colors[i - length1] + 0.5)

    return colors, weights

def get_colors_and_weights(cmap='Set1', num_colors=9, observed=True, 
    mode='Train', new_colors=None):
    """Get color array from matplotlib colormap."""
    if observed:
        c = observed_colors(num_colors, mode)
        cm = plt.get_cmap(cmap)

        colors = []
        for i in reversed(range(num_colors)):
            colors.append((cm(c[i])))

        weights = [num_colors - idx
                       for idx in range(num_colors)]
    else:
        colors, weights = unobserved_colors(cmap, num_colors, mode, new_colors)

    return colors, weights


def pairwise_distance_matrix(x, y):
    x_norm = (x**2).sum(1).view(-1, 1)
    y_norm = (y**2).sum(1).view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    return dist


def get_act_fn(act_fn):
    if act_fn == 'relu':
        return nn.ReLU(inplace=True)
    elif act_fn == 'leaky_relu':
        return nn.LeakyReLU(inplace=True)
    elif act_fn == 'elu':
        return nn.ELU()
    elif act_fn == 'sigmoid':
        return nn.Sigmoid()
    elif act_fn == 'softplus':
        return nn.Softplus()
    else:
        raise ValueError('Invalid argument for `act_fn`.')


def to_one_hot(indices, max_index):
    """Get one-hot encoding of index tensors."""
    zeros = torch.zeros(
        indices.size()[0], max_index, dtype=torch.float32,
        device=indices.device)
    #print(indices.size())
    #print(zeros.size())
    return zeros.scatter_(1, indices.unsqueeze(1), 1)


def to_float(np_array):
    """Convert numpy array to float32."""
    return np.array(np_array, dtype=np.float32)


def unsorted_segment_sum(tensor, segment_ids, num_segments):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`."""
    result_shape = (num_segments, tensor.size(1))
    result = tensor.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, tensor.size(1))
    result.scatter_add_(0, segment_ids, tensor)
    return result


class LSTMDataset(data.Dataset):

    def __init__(self, hdf5_file, action_transform=None,
                 in_memory=False):
        if action_transform is None:
            self.action_transform = lambda x:x
        else:
            self.action_transform = action_transform
        self.experience_buffer = load_list_dict_h5py(hdf5_file)
    
    def __len__(self):
        return len(self.experience_buffer)

    def __getitem__(self, idx):

        obs = []
        for o in self.experience_buffer[idx]['obs']:
            obs.append(to_float(o))

        acts = []
        for a in self.experience_buffer[idx]['action']:
            acts.append(to_float(self.action_transform(a)))

        rewards = []
        for a in self.experience_buffer[idx]['reward']:
            rewards.append(a)

        targets = []
        for t in self.experience_buffer[idx]['target']:
            targets.append(to_float(t))

        obs = np.stack(obs)
        acts = np.stack(acts)
        rewards = np.array(rewards)
        targets = np.stack(targets)

        return obs, acts, rewards, targets

class StateTransitionsDataset(data.Dataset):
    """Create dataset of (o_t, a_t, o_{t+1}) transitions from replay buffer."""

    def __init__(self, hdf5_file, action_transform=None,
                 in_memory=False):
        """
        Args:
            hdf5_file (string): Path to the hdf5 file that contains experience
                buffer
        """
        if action_transform is None:
            action_transform = lambda x: x  # noqa
        self.in_memory = in_memory
        self.action_transform = action_transform
        if in_memory:
            self.experience_buffer = h5py.File(hdf5_file, 'r')
        else:
            self.experience_buffer = load_list_dict_h5py(hdf5_file)

        # Build table for conversion between linear idx -> episode/step idx
        self.idx2episode = list()
        step = 0
        for ep in range(len(self.experience_buffer)):
            num_steps = len(self.experience_buffer[ep]['action'])
            idx_tuple = [(ep, idx) for idx in range(num_steps)]
            self.idx2episode.extend(idx_tuple)
            step += num_steps

        self.num_steps = step

    def __len__(self):
        return self.num_steps

    def __getitem__(self, idx):
        ep, step = self.idx2episode[idx]

        obs = to_float(self.experience_buffer[ep]['obs'][step])
        action = self.experience_buffer[ep]['action'][step]
        reward = self.experience_buffer[ep]['reward'][step]
        action = self.action_transform(action)
        next_obs = to_float(self.experience_buffer[ep]['next_obs'][step])
        target = to_float(self.experience_buffer[ep]['target'])

        return obs, action, next_obs, reward, target


class PathDataset(data.Dataset):
    """Create dataset of {(o_t, a_t)}_{t=1:N} paths from replay buffer.
    """

    def __init__(self, hdf5_file, *, path_length=5,
                 action_transform=None, in_memory=True):
        """
        Args:
            hdf5_file (string): Path to the hdf5 file that contains experience
                buffer
        """
        self.in_memory = in_memory
        if in_memory:
            self.experience_buffer = load_list_dict_h5py(hdf5_file)
        else:
            self.experience_buffer = h5py.File(hdf5_file, 'r')
        self.path_length = path_length
        if action_transform is None:
            action_transform = lambda x: x  # noqa
        self.action_transform = action_transform

    def __len__(self):
        return len(self.experience_buffer)

    def __getitem__(self, idx):
        observations = []
        actions = []
        for i in range(self.path_length):
            if not self.in_memory:
                idx = str(idx)
            obs = to_float(self.experience_buffer[idx]['obs'][i])
            action = self.experience_buffer[idx]['action'][i]
            action = self.action_transform(action)
            observations.append(obs)
            actions.append(action)
        obs = to_float(
            self.experience_buffer[idx]['next_obs'][self.path_length - 1])
        observations.append(obs)
        return observations, actions

def save_img(img, name, pad_value=None):
    img = img.permute(1,2,0).detach().cpu().numpy()
    plt.axis('off')
    plt.imshow(img)
    plt.savefig(name, bbox_inches='tight', pad_inches=0)

def evaluate(model, loader, *,
             device, batch_size, num_steps, silent=False, 
             name="Default", save_folder=None, contrastive=False):
    # topk = [1, 5, 10]
    name = str(name).split('/')[-1]
    topk = [1]
    hits_at = defaultdict(int)
    num_samples = 0
    rr_sum = 0
    rec = 0.0

    pred_states = []
    next_states = []

    with torch.no_grad():
        iterator = enumerate(tqdm(loader, disable=silent, leave=False))
        for batch_idx, data_batch in iterator:
            data_batch = [[t.to(device) for t in tensor]
                          for tensor in data_batch]
            observations, actions = data_batch

            obs = observations[0]
            next_obs = observations[-1]

            state,_ = model.encode(obs)
            next_state,_ = model.encode(next_obs)

            pred_state = state
            for i in range(num_steps):
                pred_state = model.transition(pred_state, actions[i])
                if model.learn_edges:
                    pred_state = pred_state[0]

            if not contrastive:
                rec_obs = model.decoder(pred_state)
                rec += image_loss(rec_obs, next_obs).item()

            pred_states.append(pred_state.cpu())
            next_states.append(next_state.cpu())

        pred_state_cat = torch.cat(pred_states, dim=0)
        next_state_cat = torch.cat(next_states, dim=0)

        full_size = pred_state_cat.size(0)

        # Flatten object/feature dimensions
        next_state_flat = next_state_cat.view(full_size, -1)
        pred_state_flat = pred_state_cat.view(full_size, -1)

        dist_matrix = pairwise_distance_matrix(
            next_state_flat, pred_state_flat)
        dist_matrix_diag = torch.diag(dist_matrix).unsqueeze(-1)
        dist_matrix_augmented = torch.cat(
            [dist_matrix_diag, dist_matrix], dim=1)

        # Workaround to get a stable sort in numpy.
        dist_np = dist_matrix_augmented.numpy()
        indices = []
        for row in dist_np:
            keys = (np.arange(len(row)), row)
            indices.append(np.lexsort(keys))
        indices = np.stack(indices, axis=0)
        indices = torch.from_numpy(indices).long()

        print('Processed {} batches of size {}'.format(
            batch_idx + 1, batch_size))

        labels = torch.zeros(
            indices.size(0), device=indices.device,
            dtype=torch.int64).unsqueeze(-1)

        num_samples += full_size
        print('Size of current top k evaluation batch: {}'.format(
            full_size))

        for k in topk:
            match = indices[:, :k] == labels
            num_matches = match.sum()
            hits_at[k] += num_matches.item()

        match = indices == labels
        _, ranks = match.max(1)

        reciprocal_ranks = torch.reciprocal(ranks.double() + 1)
        rr_sum += reciprocal_ranks.sum()

    return hits_at, rr_sum, rec, num_samples

def evaluate_lstm(model, loader, *,
             device, batch_size, num_steps, silent=False, name="Default", save_folder = None, hidden_dim = 600, contrastive=False):
    # topk = [1, 5, 10]
    name = str(name).split('/')[-1]
    topk = [1]
    hits_at = defaultdict(int)
    num_samples = 0
    rr_sum = 0
    rec = 0.0

    pred_states = []
    next_states = []

    with torch.no_grad():
        iterator = enumerate(tqdm(loader, disable=silent, leave=False))
        for batch_idx, data_batch in iterator:
            data_batch = [[t.to(device) for t in tensor]
                          for tensor in data_batch]
            observations, actions = data_batch

            obs = observations[0]
            next_obs = observations[-1]

            state,_ = model.encode(obs)
            next_state,_ = model.encode(next_obs)

            pred_state = state
            #if hidden_dim == 512:
            hidden = (torch.zeros(1, pred_state.size(0), 600).cuda(), torch.zeros(1, pred_state.size(0), 600).cuda())
            #else:
            #hidden = model.transition_nets.init_hidden(pred_state.size(0))
            for i in range(num_steps):
                pred_state, hidden = model.transition(pred_state, actions[i], hidden)

            if not contrastive:
                rec_obs = model.decoder(pred_state)
                rec += image_loss(rec_obs, next_obs).item()

            pred_states.append(pred_state.cpu())
            next_states.append(next_state.cpu())

        pred_state_cat = torch.cat(pred_states, dim=0)
        next_state_cat = torch.cat(next_states, dim=0)

        full_size = pred_state_cat.size(0)

        # Flatten object/feature dimensions
        next_state_flat = next_state_cat.view(full_size, -1)
        pred_state_flat = pred_state_cat.view(full_size, -1)

        dist_matrix = pairwise_distance_matrix(
            next_state_flat, pred_state_flat)
        dist_matrix_diag = torch.diag(dist_matrix).unsqueeze(-1)
        dist_matrix_augmented = torch.cat(
            [dist_matrix_diag, dist_matrix], dim=1)

        # Workaround to get a stable sort in numpy.
        dist_np = dist_matrix_augmented.numpy()
        indices = []
        for row in dist_np:
            keys = (np.arange(len(row)), row)
            indices.append(np.lexsort(keys))
        indices = np.stack(indices, axis=0)
        indices = torch.from_numpy(indices).long()

        print('Processed {} batches of size {}'.format(
            batch_idx + 1, batch_size))

        labels = torch.zeros(
            indices.size(0), device=indices.device,
            dtype=torch.int64).unsqueeze(-1)

        num_samples += full_size
        print('Size of current top k evaluation batch: {}'.format(
            full_size))

        for k in topk:
            match = indices[:, :k] == labels
            num_matches = match.sum()
            hits_at[k] += num_matches.item()

        match = indices == labels
        _, ranks = match.max(1)

        reciprocal_ranks = torch.reciprocal(ranks.double() + 1)
        rr_sum += reciprocal_ranks.sum()

    return hits_at, rr_sum, rec, num_samples


def eval_steps(model, steps_array, *,
               filename, batch_size, device, save_folder, 
               name="Default", action_dim = 5, contrastive=False):
    string = ""
    for steps in steps_array:
        print("Loading data...")
        
        dataset = PathDataset(
            hdf5_file=filename, path_length=steps,
            action_transform=OneHot(model.num_objects * action_dim),
            in_memory=False)
        
        eval_loader = data.DataLoader(
            dataset, batch_size=batch_size, 
            shuffle=False, num_workers=0)

        hits_at, rr_sum, rec, num_samples = evaluate(
            model, eval_loader,
            device=device, batch_size=batch_size,
            num_steps=steps, name = name + str(steps), 
            save_folder=save_folder, contrastive=contrastive)

        print(f'Steps: {steps}')
        for k in [1]:
            print(f'H@{k}: {hits_at[k] / num_samples * 100:.2f}%')

        print(f'MRR: {rr_sum / num_samples * 100:.2f}%')
        print(f'Reconstruction: {rec / num_samples :.2f}')
        string = string + f'{hits_at[k] / num_samples * 100:.2f} & {rr_sum / num_samples * 100:.2f} & {rec / num_samples :.2f} & '

    with open(str(save_folder) + '/eval.txt', 'a') as f:
        f.write(f'{name} : {string} \n')

    print(string)

def eval_steps_lstm(model, steps_array, *,
               filename, batch_size, device, save_folder, name="Default", action_dim = 5, hidden_dim = 600, contrastive=False):
    string = ""
    for steps in steps_array:
        print("Loading data...")
        dataset = PathDataset(
            hdf5_file=filename, path_length=steps,
            action_transform=OneHot(model.num_objects * action_dim),
            in_memory=False)
        eval_loader = data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        hits_at, rr_sum, rec, num_samples = evaluate_lstm(
            model, eval_loader,
            device=device, batch_size=batch_size,
            num_steps=steps, name = name + str(steps), save_folder = save_folder, hidden_dim = hidden_dim)

        print(f'Steps: {steps}')
        for k in [1]:
            print(f'H@{k}: {hits_at[k] / num_samples * 100:.2f}%')

        print(f'MRR: {rr_sum / num_samples * 100:.2f}%')
        print(f'Reconstruction: {rec / num_samples :.2f}')
        string = string + f'{hits_at[k] / num_samples * 100:.2f} & {rr_sum / num_samples * 100:.2f} & {rec / num_samples :.2f} & '

    with open(str(save_folder) + '/eval.txt', 'a') as f:
        f.write(f'{name} : {string} \n')

    print(string)
