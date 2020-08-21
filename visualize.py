import argparse
import torch
import torch.nn.functional as F
import pickle
from pathlib import Path

from torch.utils import data
import numpy as np
import tqdm
import logging

from cswm import utils
from cswm.models.modules import RewardPredictor, CausalTransitionModel#, ContrastiveSWM
from cswm.utils import OneHot, PathDataset

import sys
import datetime
import os
import pickle
from pathlib import Path

import numpy as np

from itertools import chain
from torchvision.utils import save_image
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('--save-folder', type=Path,
                    default='checkpoints',
                    help='Path to checkpoints.')
args_eval = parser.parse_args()


meta_file = args_eval.save_folder / 'metadata.pkl'
model_file = args_eval.save_folder / 'finetuned_model.pt'

with open(meta_file, 'rb') as f:
    args = pickle.load(f)['args']



np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
device = torch.device('cuda' if args.cuda else 'cpu')


save_folder = args_eval.save_folder 
input_shape = (3, 50, 50)



model = CausalTransitionModel(
        embedding_dim_per_object=args.embedding_dim_per_object,
        hidden_dim=args.hidden_dim,
        action_dim=args.action_dim,
        input_dims=input_shape,
        input_shape=input_shape,
        modular=args.modular,
        predict_diff=args.predict_diff,
        vae=args.vae,
        num_objects=args.num_objects,
        encoder=args.encoder,
        gnn=args.gnn,
        multiplier=args.multiplier,
        ignore_action=args.ignore_action,
        copy_action=args.copy_action).to(device)

model.load_state_dict(torch.load(model_file))

dataset = PathDataset(
            hdf5_file=args.eval_dataset, path_length=10,
            action_transform=OneHot(args.num_objects * args.action_dim),
            in_memory=False)

obs, actions = dataset[50]

for i in range(len(obs)):
    obs[i] = torch.from_numpy(obs[i]).cuda()
    
for i in range(len(actions)):
    actions[i] = torch.from_numpy(actions[i]).cuda()


save_image(torch.stack(obs, dim = 0), save_folder / 'gt.png', pad_value = 1.0, nrow = 1)
obs_pred = []
state, _ = model.encode(obs[0].cuda().unsqueeze(0))
ops = []
for i in range(len(obs)):
    rec = torch.sigmoid(model.decoder(state))
    ops.append(rec.squeeze(0))
    if i <  len(obs) - 1:
        state = model.transition(state, actions[i].cuda().unsqueeze(0))
        #state, _ = model.encode(obs[i+1].cuda().unsqueeze(0))
save_image(torch.stack(ops, dim = 0), save_folder / 'pred.png', pad_value = 1.0, nrow = 1)

obs_pred = []
state, _ = model.encode(obs[0].cuda().unsqueeze(0))
ops = []
for i in range(len(obs)):
    rec = torch.sigmoid(model.decoder(state))
    ops.append(rec.squeeze(0))
    if i <  len(obs) - 1:
        #state = model.transition(state, actions[i].cuda().unsqueeze(0))
        state, _ = model.encode(obs[i+1].cuda().unsqueeze(0))
save_image(torch.stack(ops, dim = 0), save_folder / 'predrec.png', pad_value = 1.0, nrow = 1)
