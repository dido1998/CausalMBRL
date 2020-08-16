import argparse
import torch
import datetime
import os
import pickle
import tqdm
import sys
from pathlib import Path

import numpy as np
import logging
import re

from itertools import chain
from torch.utils import data
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from cswm import utils
from cswm.models.modules import CausalTransitionModel, ContrastiveSWM, ContrastiveSWMFinal
from cswm.models.losses import *

from cswm.utils import OneHot


parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=1024,
                    help='Batch size.')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of training epochs.')
parser.add_argument('--pretrain-epochs', type=int, default=100,
                    help='Number of pretraining epochs.')
parser.add_argument('--lr', type=float, default=5e-4,
                    help='Learning rate.')
parser.add_argument('--transit-lr', type=float, default=5e-4,
                    help='Learning rate for transition model.')
parser.add_argument('--update-interval', type=int, default=10,
                    help='update interval for structural params.')
parser.add_argument('--encoder', type=str, default='small',
                    help='Object extractor CNN size (e.g., `small`).')
parser.add_argument('--multiplier', type=int, default=1)

parser.add_argument('--sigma', type=float, default=0.5,
                    help='Energy scale.')
parser.add_argument('--hinge', type=float, default=1.,
                    help='Hinge threshold parameter.')
parser.add_argument('--modular', action='store_true',
                    help='Is the learned model modular?')
parser.add_argument('--vae', action='store_true',
                    help='Is the learned encoder decoder model a VAE model?')
parser.add_argument('--predict-diff', action='store_true',
                    help='Do we predict the difference of current and next state?')
parser.add_argument('--hidden-dim', type=int, default=512,
                    help='Number of hidden units in transition MLP.')
parser.add_argument('--embedding-dim-per-object', type=int, default=5,
                    help='Dimensionality of embedding.')
parser.add_argument('--action-dim', type=int, default=5,
                    help='Dimensionality of action space.')
parser.add_argument('--num-objects', type=int, default=5,
                    help='Number of object slots in model.')
parser.add_argument('--ignore-action', action='store_true', default=False,
                    help='Ignore action in GNN transition model.')
parser.add_argument('--copy-action', action='store_true', default=False,
                    help='Apply same action to all object slots.')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disable CUDA training.')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed (default: 42).')
parser.add_argument('--log-interval', type=int, default=5,
                    help='How many batches to wait before logging'
                         'training status.')
parser.add_argument('--silent', action='store_true',
                    help='When selected, the progress bar is not shown')

# Dataset

parser.add_argument('--dataset', type=Path,
                    default=Path('data/weighted_shapes_train.h5'),
                    help='Path to replay buffer.')
parser.add_argument('--valid-dataset', type=Path,
                    default=Path('data/weighted_shapes_valid.h5'),
                    help='Path to replay buffer.')
parser.add_argument('--eval-dataset', type=Path,
                    default=None,
                    help='Path to replay buffer.')
parser.add_argument('--num-workers', type=int, default=1,
                    help='Number of data loading workers')

parser.add_argument('--contrastive-loss', type=bool, default=True,
                    help="whether to use contrastive loss")
parser.add_argument('--name', type=str, default='none',
                    help='Experiment name.')
parser.add_argument('--save-folder', type=Path,
                    default=Path('checkpoints'),
                    help='Path to checkpoints.')
parser.add_argument('--gnn', action = 'store_true', help='use GNN model (Kipf et al)')
parser.add_argument('--contrastive', action = 'store_true', help='use contrastive loss')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# Set experiment name

now = datetime.datetime.now()
timestamp = now.isoformat()

if args.name == 'none':
    exp_name = timestamp
else:
    exp_name = args.name

save_folder = args.save_folder / exp_name

if not os.path.exists(save_folder):
    os.makedirs(save_folder)
else:
    raise ValueError(f'Save folder already exists: {save_folder}')

meta_file = save_folder / 'metadata.pkl'
model_file = save_folder / 'model.pt'
finetune_file = save_folder / 'finetuned_model.pt'

log_file = save_folder / 'log.txt'

# Set seeds

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Set logging

handlers = [logging.FileHandler(log_file, 'a')]
if args.silent:
    handlers.append(logging.StreamHandler(sys.stdout))
logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=handlers)
logger = logging.getLogger()
print = logger.info

with open(meta_file, "wb") as f:
    pickle.dump({'args': args}, f)

device = torch.device('cuda' if args.cuda else 'cpu')

# Load datasets

dataset = utils.StateTransitionsDataset(
    hdf5_file=args.dataset, action_transform=OneHot(args.num_objects * args.action_dim))
valid_dataset = utils.StateTransitionsDataset(
    hdf5_file=args.valid_dataset, action_transform=OneHot(args.num_objects * args.action_dim))

train_loader = data.DataLoader(
    dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
valid_loader = data.DataLoader(
    valid_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

# Get data sample
obs = next(iter(train_loader))[0]
input_shape = obs[0].size()

# Initialize Model

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
    sigma=args.sigma,
    hinge=args.hinge,
    ignore_action=args.ignore_action,
    copy_action=args.copy_action).to(device)

num_enc = sum(p.numel() for p in model.encoder_parameters())
num_dec = sum(p.numel() for p in model.decoder_parameters())
num_tr = sum(p.numel() for p in model.transition_parameters())

print(f'Number of parameters in Encoder: {num_enc}')
print(f'Number of parameters in Decoder: {num_dec}')
print(f'Number of parameters in Transition: {num_tr}')
print(f'Number of parameters: {num_enc+num_dec+num_tr}')

model.apply(utils.weights_init)

def evaluate(model_file, valid_loader, train_encoder = True, train_decoder = True, train_transition = False):
    model.eval()
    valid_loss = 0.0

    with torch.no_grad():
        for batch_idx, data_batch in enumerate(valid_loader):
            data_batch = [tensor.to(device) for tensor in data_batch]
            obs, action, next_obs, _, _ = data_batch

            loss = 0.0

            if args.contrastive:
                state, _ = model.encode(obs)
                next_state, _ = model.encode(next_obs)
                pred_state = model.transition(state, action)
                loss = contrastive_loss(state, action, next_state, pred_state,
                                        hinge=args.hinge, sigma=args.sigma)
            else:
                state, mean_var = model.encode(obs)
                next_state, next_mean_var = model.encode(next_obs)
                pred_state = model.transition(state, action)
                recon = model.decoder(state)
                next_recon = model.decoder(next_state)

                if train_encoder and train_decoder:
                    loss += 0.5 * (image_loss(recon, obs) + image_loss(next_recon, next_obs))
                    if args.vae:
                        loss += 0.5 * (kl_loss(mean_var[0], mean_var[1]) + kl_loss(next_mean_var[0], next_mean_var[1]))
                if train_transition:
                    loss += transition_loss(pred_state, next_state)

                loss /= obs.size(0)

            valid_loss += loss.item()
        avg_loss = valid_loss / len(valid_loader)
        print('====> Average valid loss: {:.6f}'.format(avg_loss))
        return avg_loss

def train(max_epochs, model_file, lr, train_encoder=True, train_decoder=True,
          train_transition=False):

    parameters = []
    if train_decoder:
        parameters = chain(parameters, model.decoder_parameters())
    if train_encoder:
        parameters = chain(parameters, model.encoder_parameters())
    if train_transition:
        parameters = chain(parameters, model.transition_parameters())

    optimizer = torch.optim.Adam(parameters, lr = lr)

    print('Starting model training...')
    best_loss = 1e9

    for epoch in range(1, max_epochs + 1):
        train_loss = 0

        iterator = tqdm.tqdm(train_loader, desc=f'Epoch {epoch}',
                             disable=args.silent)
        for batch_idx, data_batch in enumerate(iterator):
            model.train()
            data_batch = [tensor.to(device) for tensor in data_batch]
            obs, action, next_obs, _, _ = data_batch

            optimizer.zero_grad()

            loss = 0.0

            if args.contrastive:
                state, _ = model.encode(obs)
                next_state, _ = model.encode(next_obs)
                pred_state = model.transition(state, action)
                loss = contrastive_loss(state, action, next_state, pred_state,
                                        hinge=args.hinge, sigma=args.sigma)
            else:
                state, mean_var = model.encode(obs)
                next_state, next_mean_var = model.encode(next_obs)
                pred_state = model.transition(state, action)
                recon = model.decoder(state)
                next_recon = model.decoder(next_state)

                if train_encoder and train_decoder:
                    loss += 0.5 * (image_loss(recon, obs) + image_loss(next_recon, next_obs))
                    if args.vae:
                        loss += 0.5 * (kl_loss(mean_var[0], mean_var[1]) + kl_loss(next_mean_var[0], next_mean_var[1]))
                if train_transition:
                    loss += transition_loss(pred_state, next_state)

                loss /= obs.size(0)

            loss.backward()
            train_loss += loss.item()

            optimizer.step()

            if batch_idx % args.log_interval == 0 and batch_idx > 0:
                iterator.set_postfix(loss=f'{train_loss / (1 + batch_idx):.6f}')
                print(
                    'Epoch: {} [ {}/{} ] \t Loss: {:.6f}'.format(
                        epoch, (batch_idx+1),
                        len(train_loader.dataset),
                        loss.item()))

        avg_loss = train_loss / len(train_loader)
        print('====> Epoch: {} Average train loss: {:.6f}'.format(
            epoch, avg_loss))

        avg_loss = evaluate(model_file, valid_loader, train_encoder = train_encoder, train_decoder = train_decoder, train_transition = train_transition)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), model_file)

if args.contrastive:
    train(args.epochs, model_file, lr=args.lr, train_encoder=True, train_transition=True, train_decoder=False)
else:
    train(args.pretrain_epochs, model_file, lr=args.lr, train_encoder=True, train_transition=False, train_decoder=True)
    train(args.epochs, model_file, lr=args.transit_lr, train_encoder=False, train_transition=True, train_decoder=False)
    train(args.epochs, finetune_file, lr=args.lr, train_encoder=True, train_decoder=True, train_transition=True)

if args.eval_dataset is not None:
    utils.eval_steps(
        model, [1, 5, 10],
        filename=args.eval_dataset, batch_size=args.batch_size,
        save_folder = save_folder, device=device, action_dim = args.action_dim, cswm  = args.cswm)
