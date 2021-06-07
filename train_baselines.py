import argparse
import torch
import datetime
import os
import pickle
import tqdm
import sys
from pathlib import Path
import random

import numpy as np
import logging
import re

import torch
from itertools import chain
from torch.utils import data
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from cswm import utils
from cswm.models.modules import CausalTransitionModel
from cswm.models.losses import *

from cswm.utils import OneHot


parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=1024,
                    help='Batch size.')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of training epochs.')
parser.add_argument('--pretrain-epochs', type=int, default=100,
                    help='Number of pretraining epochs.')
parser.add_argument('--finetune-epochs', type=int, default=100,
                    help='Number of finetune epochs.')
parser.add_argument('--slr', type=float, default=5e-4,
                    help='Structural learning rate.')
parser.add_argument('--lr', type=float, default=5e-4,
                    help='Learning rate.')
parser.add_argument('--transit-lr', type=float, default=5e-4,
                    help='Learning rate for transition model.')
parser.add_argument('--finetune-lr', type=float, default=5e-4,
                    help='Learning rate for finetune model.')
parser.add_argument('--update-interval', type=int, default=10,
                    help='update interval for structural params.')
parser.add_argument('--encoder', type=str, default='small',
                    help='Object extractor CNN size (e.g., `small`).')
parser.add_argument('--multiplier', type=int, default=1)
parser.add_argument('--lsparse', type=float, default=0.01)
parser.add_argument('--sigma', type=float, default=0.5,
                    help='Energy scale.')
parser.add_argument('--hinge', type=float, default=1.,
                    help='Hinge threshold parameter.')
parser.add_argument('--modular', action='store_true',
                    help='Is the learned model modular?')
parser.add_argument('--causal', action='store_true',
                    help='Is the learned model causal?')
parser.add_argument('--vae', action='store_true',
                    help='Is the learned encoder decoder model a VAE model?')
parser.add_argument('--learn-edges', action='store_true')
parser.add_argument('--predict-diff', action='store_true',
                    help='Do we predict the difference of current and next state?')
parser.add_argument('--hidden-dim', type=int, default=512,
                    help='Number of hidden units in transition MLP.')
parser.add_argument('--embedding-dim-per-object', type=int, default=5,
                    help='Dimensionality of embedding.')
parser.add_argument('--num-graphs', type = int, default = 10)
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
parser.add_argument('--graph', type=Path,
                    default=Path('data/ColorChangingRL_3-3-10-Static-train-graph-chain3'),
                    help='Path to graph.')
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

parser.add_argument('--name', type=str, default='none',
                    help='Experiment name.')
parser.add_argument('--save-folder', type=Path,
                    default=Path('checkpoints'),
                    help='Path to checkpoints.')
parser.add_argument('--gnn', action = 'store_true', help='use GNN model (Kipf et al)')
parser.add_argument('--contrastive', action = 'store_true', help='use contrastive loss')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

def set_seed(seed):
    """Set seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.cuda:
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True

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

set_seed(args.seed)
sigmoid = nn.Sigmoid()

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
graph = None #torch.load(args.graph)['graph']
dataset = utils.StateTransitionsDataset(
    hdf5_file=args.dataset, action_transform=OneHot(args.num_objects * args.action_dim))
valid_dataset = utils.StateTransitionsDataset(
    hdf5_file=args.valid_dataset, action_transform=OneHot(args.num_objects * args.action_dim))

train_loader = data.DataLoader(
    dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
valid_loader = data.DataLoader(
    valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

# Get data sample
obs = next(iter(train_loader))[0]
input_shape = obs[0].size()

# Initialize Model
#learn_edges = args.learn_edges or args.causal

model = CausalTransitionModel(
    embedding_dim_per_object=args.embedding_dim_per_object,
    hidden_dim=args.hidden_dim,
    action_dim=args.action_dim,
    input_dims=input_shape,
    input_shape=input_shape,
    modular=args.modular,
    learn_edges = args.learn_edges,
    causal = args.causal,
    predict_diff=args.predict_diff,
    vae=args.vae,
    num_graphs = args.num_graphs,
    num_objects=args.num_objects,
    encoder=args.encoder,
    gnn=args.gnn,
    graph=graph,
    multiplier=args.multiplier,
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

def eval_all_loss():
    if args.eval_dataset is not None:
        utils.eval_steps(
                model, [1, 5, 10],
                filename=args.eval_dataset, batch_size=args.batch_size,
                save_folder = save_folder, device=device, action_dim = args.action_dim, contrastive = args.contrastive)


def evaluate(model_file, valid_loader, train_encoder = True, train_decoder = True, train_transition = False):
    model.eval()
    valid_loss = 0.0


    with torch.no_grad():
        for batch_idx, data_batch in enumerate(valid_loader):
            data_batch = [tensor.to(device) for tensor in data_batch]
            obs, action, next_obs, _, _ = data_batch
            loss = 0.0

            state, mean_var = model.encode(obs)
            next_state, next_mean_var = model.encode(next_obs)
            if args.learn_edges:
                pred_state, pred_states, gamma_exp = model.transition(state, action)
            else:
                pred_state = model.transition(state, action)

            if args.contrastive:
                loss = contrastive_loss(state, action, next_state, pred_state,
                                        hinge=args.hinge, sigma=args.sigma)
            else:
                recon = model.decoder(state)
                next_recon = model.decoder(pred_state)
                #next_recon = model.decoder(next_state)

                if train_encoder and train_decoder:
                    loss += image_loss(recon, obs)
                    if args.vae:
                        loss += kl_loss(mean_var[0], mean_var[1])
                    if train_transition:
                        loss += image_loss(next_recon, next_obs)
                        if args.vae:
                            loss += kl_loss(next_mean_var[0], next_mean_var[1])
                
                if train_transition:
                    if args.learn_edges:
                        loss_, dRdgamma = causal_loss(pred_states, next_state, gamma_exp, model.gamma)
                        loss += loss_
                    else:
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
    struct_optimizer = torch.optim.Adam(
        model.structural_parameters(),
        lr=args.slr)

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

            state, mean_var = model.encode(obs)
            next_state, next_mean_var = model.encode(next_obs)
            if args.learn_edges:
                pred_state, pred_states, gamma_exp = model.transition(state, action)
            else:
                pred_state = model.transition(state, action)
            if args.contrastive:
                loss = contrastive_loss(state, action, next_state, pred_state,
                                        hinge=args.hinge, sigma=args.sigma)
            else:
                recon = model.decoder(state)
                next_recon = model.decoder(pred_state)
                #next_recon = model.decoder(next_state)

                if train_encoder and train_decoder:
                    loss += image_loss(recon, obs)
                    if args.vae:
                        loss += kl_loss(mean_var[0], mean_var[1])
                    if train_transition:
                        loss += image_loss(next_recon, next_obs)
                        if args.vae:
                            loss += kl_loss(next_mean_var[0], next_mean_var[1])
                
                if train_transition:
                    if args.learn_edges:
                        loss_, dRdgamma = causal_loss(pred_states, next_state, gamma_exp, model.gamma)
                        l1_loss = sigmoid(model.gamma).sum()
                        loss += loss_ + args.lsparse * l1_loss
                    else:
                        loss += transition_loss(pred_state, next_state)

                loss /= obs.size(0)


            loss.backward()
            
            
            train_loss += loss.item()

            if train_transition and args.learn_edges:
                if (batch_idx // args.update_interval) % 2 ==0:
                    # udpate functional parameters only
                    struct_optimizer.zero_grad()
                else:
                    optimizer.zero_grad()
                    # manually define grad for gamma params
                    model.gamma.grad = torch.zeros_like(model.gamma)
                    model.gamma.grad.copy_(dRdgamma)
                    struct_optimizer.step()

            optimizer.step()

            if batch_idx % args.log_interval == 0 and batch_idx > 0:
                iterator.set_postfix(loss=f'{train_loss / (1 + batch_idx):.6f}')
                print(
                    'Epoch: {} [ {}/{} ] \t Loss: {:.6f}'.format(
                        epoch, (batch_idx+1),
                        len(train_loader.dataset),
                        loss.item()))
                if train_transition and args.learn_edges:
                    print(sigmoid(model.gamma))

        avg_loss = train_loss / len(train_loader)
        print('====> Epoch: {} Average train loss: {:.6f}'.format(
            epoch, avg_loss))

        avg_loss = evaluate(model_file, valid_loader, train_encoder = train_encoder, train_decoder = train_decoder, train_transition = train_transition)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), model_file)
        
        if epoch > 0 and epoch % 100 == 0:
            eval_all_loss()
            

if args.contrastive:
    train(args.epochs, model_file, lr=args.lr, train_encoder=True, train_transition=True, train_decoder=False)
else:
    train(args.pretrain_epochs, model_file, lr=args.lr, train_encoder=True, train_transition=False, train_decoder=True)
    train(args.epochs, model_file, lr=args.transit_lr, train_encoder=False, train_transition=True, train_decoder=False)
    train(args.finetune_epochs, finetune_file, lr=args.finetune_lr, train_encoder=True, train_decoder=True, train_transition=True)

#if args.eval_dataset is not None:
#    utils.eval_steps(
#        model, [1, 5, 10],
#        filename=args.eval_dataset, batch_size=args.batch_size,
#        import pdb; pdb.set_trace()
#        save_folder = save_folder, device=device, action_dim = args.action_dim, contrastive = args.contrastive)
