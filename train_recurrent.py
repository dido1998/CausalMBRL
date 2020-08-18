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
import torch.nn.functional as F
import torchvision

from cswm import utils
from cswm.models.modules import CausalTransitionModelLSTM
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
parser.add_argument('--s-lr', type=float, default=5e-3,
                    help='Learning rate.')
parser.add_argument('--update-interval', type=int, default=10,
                    help='update interval for structural params.')
parser.add_argument('--encoder', type=str, default='small',
                    help='Object extractor CNN size (e.g., `small`).')

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
parser.add_argument('--embedding-dim-per-object', type=int, default=2,
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

parser.add_argument('--contrastive', action='store_true', default=False,
                    help="whether to use contrastive loss")
parser.add_argument('--name', type=str, default='none',
                    help='Experiment name.')
parser.add_argument('--save-folder', type=Path,
                    default=Path('checkpoints'),
                    help='Path to checkpoints.')
parser.add_argument('--rim', action = 'store_true')
parser.add_argument('--scoff', action = 'store_true')
parser.add_argument('--rules', action = 'store_true')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

now = datetime.datetime.now()
timestamp = now.isoformat()

if args.name == 'none':
    exp_name = timestamp
else:
    exp_name = args.name

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

exp_counter = 0
save_folder = args.save_folder / exp_name

if not os.path.exists(save_folder):
    os.makedirs(save_folder)
else:
    raise ValueError(f'Save folder already exists: {save_folder}')

meta_file = save_folder / 'metadata.pkl'
model_file = save_folder / 'model.pt'
finetune_file = save_folder / 'finetuned_model.pt'

log_file = save_folder / 'log.txt'

handlers = [logging.FileHandler(log_file, 'a')]
if args.silent:
    handlers.append(logging.StreamHandler(sys.stdout))
logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=handlers)
logger = logging.getLogger()
print = logger.info

with open(meta_file, "wb") as f:
    pickle.dump({'args': args}, f)

device = torch.device('cuda' if args.cuda else 'cpu')

model = CausalTransitionModelLSTM(
    embedding_dim_per_object=args.embedding_dim_per_object,
    hidden_dim=args.hidden_dim,
    action_dim=args.action_dim,
    input_dims=(3, 50, 50),
    input_shape=(3, 50, 50),
    modular=args.modular,
    predict_diff=args.predict_diff,
    vae=args.vae,
    num_objects=args.num_objects,
    encoder=args.encoder, 
    rim = args.rim,
    scoff = args.scoff).to(device)

model.apply(utils.weights_init)

def evaluate(model_file, valid_loader, train_encoder = True, train_decoder = True, train_transition = False):
    model.eval()
    valid_loss = 0.0

    with torch.no_grad():
        for batch_idx, data_batch in enumerate(valid_loader):
            data_batch = [tensor.to(device) for tensor in data_batch]
            if train_transition:
                obs, action, _, _ = data_batch
                obs = obs.transpose(1,0)
                action = action.transpose(1,0)
            else:
                obs, action, next_obs, _, _ = data_batch
                obs = torch.stack([obs, next_obs])
                action = action.unsqueeze(0)

            hidden = (torch.zeros(1, obs[0].size(0), 600).cuda(), 
                torch.zeros(1, obs[0].size(0), 600).cuda())

            loss = 0.0
            n_examples = 0

            for j in range(obs.shape[0] - 1):
                state, mean_var = model.encode(obs[j])
                next_state, next_mean_var = model.encode(obs[j+1])
                pred_state, hidden = model.transition(state, action[j], hidden)

                if args.contrastive:
                    loss += contrastive_loss(state, action[j], next_state, pred_state,
                                            args.hinge, args.sigma)
                else:
                    recon = model.decoder(state)
                    next_recon = model.decoder(next_state)

                    if train_encoder and train_decoder:
                        loss += image_loss(recon, obs[j])
                        if args.vae:
                            loss += kl_loss(mean_var[0], mean_var[1])
                        if train_transition:
                            loss += image_loss(next_recon, obs[j+1])
                            if args.vae:
                                loss += kl_loss(next_mean_var[0], next_mean_var[1])
                    
                    if train_transition:
                        loss += transition_loss(pred_state, next_state)

                n_examples += obs[j].squeeze(0).size(0)

            loss /= float(n_examples)            
            valid_loss += loss.item()

        avg_loss = valid_loss / len(valid_loader)
        print('====> Average valid loss: {:.6f}'.format(avg_loss))
        return avg_loss

def train(max_epochs, model_file, lr, train_encoder=True, train_decoder=True,
          train_transition=False, train_gamma=False):

    parameters = []
    if train_transition:
        parameters = chain(parameters, model.transition_parameters())
    if train_encoder:
        parameters = chain(parameters, model.encoder_parameters())
    if train_decoder:
        parameters = chain(parameters, model.decoder_parameters())

    optimizer = torch.optim.Adam(parameters, lr=lr)

    print('Starting model training...')
    best_loss = 1e9
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0

        iterator = tqdm.tqdm(train_loader, desc=f'Epoch {epoch}',
                             disable=args.silent)
        for batch_idx, data_batch in enumerate(iterator):
            model.train()
            data_batch = [tensor.to(device) for tensor in data_batch]

            if train_transition:
                obs, action, _, _ = data_batch
                obs = obs.transpose(1,0)
                action = action.transpose(1,0)
            else:
                obs, action, next_obs, _, _ = data_batch
                obs = torch.stack([obs, next_obs])
                action = action.unsqueeze(0)

            optimizer.zero_grad()

            hidden = (torch.zeros(1, obs[0].size(0), 600).cuda(), 
                torch.zeros(1, obs[0].size(0), 600).cuda())

            loss = 0.0
            n_examples = 0

            for j in range(obs.shape[0] - 1):
                state, mean_var = model.encode(obs[j])
                next_state, next_mean_var = model.encode(obs[j+1])
                pred_state, hidden = model.transition(state, action[j], hidden)

                if args.contrastive:
                    loss += contrastive_loss(state, action[j], next_state, pred_state,
                                            args.hinge, args.sigma)
                else:
                    recon = model.decoder(state)
                    next_recon = model.decoder(next_state)

                    if train_encoder and train_decoder:
                        loss += image_loss(recon, obs[j])
                        if args.vae:
                            loss += kl_loss(mean_var[0], mean_var[1])
                        if train_transition:
                            loss += image_loss(next_recon, obs[j+1])
                            if args.vae:
                                loss += kl_loss(next_mean_var[0], next_mean_var[1])
                    
                    if train_transition:
                        loss += transition_loss(pred_state, next_state)

                n_examples += obs[j].squeeze(0).size(0)

            loss /= float(n_examples)
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
        print('====> Epoch: {} Average loss: {:.6f}'.format(
            epoch, avg_loss))

        # Add Validation
        avg_loss = evaluate(model_file, valid_loader, train_encoder = train_encoder, train_decoder = train_decoder, train_transition = train_transition)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), model_file)

if args.contrastive:
    dataset = utils.LSTMDataset(
        hdf5_file=args.dataset, action_transform=OneHot(args.num_objects * args.action_dim))
    train_loader = data.DataLoader(
        dataset, batch_size=32, shuffle=True, num_workers=args.num_workers)
    valid_dataset = utils.LSTMDataset(
        hdf5_file=args.valid_dataset, action_transform=OneHot(args.num_objects * args.action_dim))
    valid_loader = data.DataLoader(
        valid_dataset, batch_size=32, shuffle=True, num_workers=args.num_workers)

    train(args.epochs, model_file, lr=args.lr, train_encoder=True, train_transition=True, train_decoder=False)
else:
    dataset = utils.StateTransitionsDataset(
        hdf5_file=args.dataset, action_transform=OneHot(args.num_objects * args.action_dim))
    train_loader = data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_dataset = utils.StateTransitionsDataset(
        hdf5_file=args.valid_dataset, action_transform=OneHot(args.num_objects * args.action_dim))
    valid_loader = data.DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    train(args.pretrain_epochs, model_file, lr=args.lr, train_encoder=True, train_transition=False, train_decoder=True)

    del dataset, train_loader, valid_loader, valid_dataset
    dataset = utils.LSTMDataset(
        hdf5_file=args.dataset, action_transform=OneHot(args.num_objects * args.action_dim))
    train_loader = data.DataLoader(
        dataset, batch_size=32, shuffle=True, num_workers=args.num_workers)
    valid_dataset = utils.LSTMDataset(
        hdf5_file=args.valid_dataset, action_transform=OneHot(args.num_objects * args.action_dim))
    valid_loader = data.DataLoader(
        valid_dataset, batch_size=32, shuffle=True, num_workers=args.num_workers)

    train(args.epochs, model_file, lr=args.transit_lr, train_encoder=False, train_transition=True, train_decoder=False)
    train(args.epochs, finetune_file, lr=args.lr, train_encoder=True, train_decoder=True, train_transition=True)

if args.eval_dataset is not None:
    utils.eval_steps_lstm(
        model, [1, 5, 10],
        filename=args.eval_dataset, batch_size=args.batch_size,
        save_folder = save_folder, device=device, 
        action_dim = args.action_dim, hidden_dim = args.hidden_dim)