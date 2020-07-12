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
from cswm.models.modules import CausalTransitionModel, ContrastiveSWM

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
parser.add_argument('--multiplier', type=int, default=1)

parser.add_argument('--sigma', type=float, default=0.5,
                    help='Energy scale.')
parser.add_argument('--hinge', type=float, default=1.,
                    help='Hinge threshold parameter.')
parser.add_argument('--modular', action='store_true',
                    help='Is the learned model modular?')
parser.add_argument('--vae', action='store_true',
                    help='Is the learned encoder decoder model a VAE model?')
parser.add_argument('--learn-edges', action='store_true',
                    help='Does the model have learned edges?')
parser.add_argument('--predict-diff', action='store_true',
                    help='Do we predict the difference of current and next state?')
parser.add_argument('--hidden-dim', type=int, default=512,
                    help='Number of hidden units in transition MLP.')
parser.add_argument('--embedding-dim', type=int, default=5,
                    help='Dimensionality of embedding.')
parser.add_argument('--action-dim', type=int, default=4,
                    help='Dimensionality of action space.')
parser.add_argument('--num-objects', type=int, default=5,
                    help='Number of object slots in model.')
parser.add_argument('--num-graphs', type=int, default=10,
                    help='Number of graphs to sample.')
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
parser.add_argument('--reload-folder', type=Path,
                    default=Path('checkpoints'),
                    help='Path to reload file.')
parser.add_argument('--reload', action='store_true',
                    help='reload encoder, decoder')
parser.add_argument('--cswm', action = 'store_true', help='use c-swm model (Kipf et al)')


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
reload_file = args.reload_folder / 'model.pt'

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

if args.cswm:
    model = ContrastiveSWM(
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        action_dim=args.action_dim,
        input_dims=input_shape,
        num_objects=args.num_objects,
        sigma=args.sigma,
        hinge=args.hinge,
        ignore_action=args.ignore_action,
        copy_action=args.copy_action,
        encoder=args.encoder).to(device)

    num_enc = sum(p.numel() for p in model.encoder_parameters())
    num_tr = sum(p.numel() for p in model.transition_parameters())
    print(f'Number of parameters in Encoder: {num_enc}')
    print(f'Number of parameters in Transition: {num_tr}')
    print(f'Number of parameters: {num_enc + num_tr}')
else:
    model = CausalTransitionModel(
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        action_dim=args.action_dim,
        input_dims=input_shape,
        input_shape=input_shape,
        num_graphs=args.num_graphs,
        modular=args.modular,
        predict_diff=args.predict_diff,
        learn_edges=args.learn_edges,
        vae=args.vae,
        num_objects=args.num_objects,
        encoder=args.encoder,
        multiplier=args.multiplier).to(device)
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

    for batch_idx, data_batch in enumerate(valid_loader):
        data_batch = [tensor.to(device) for tensor in data_batch]
        obs, action, next_obs = data_batch

        loss = 0.0
        if not args.cswm:
            state, kl_loss = model.encode(obs)

            if train_encoder or train_decoder:
                rec_state = torch.sigmoid(model.decoder(state))
                loss += (F.binary_cross_entropy(
                    rec_state, obs, reduction='sum') + kl_loss)
            if train_transition:
                next_state, _ = model.encode(next_obs)

                transition_loss, _, pred_next = model.transition(
                    state, action, next_state)
                loss += transition_loss

                if train_encoder and train_decoder:
                    loss += F.binary_cross_entropy(torch.sigmoid(model.decoder(pred_next)), next_obs,
                                    reduction='sum')
        else:
            loss = model.contrastive_loss(*data_batch)
        loss /= obs.size(0)

        valid_loss += loss.item()
    avg_loss = valid_loss / len(valid_loader)
    print('====> Average valid loss: {:.6f}'.format(avg_loss))
    return avg_loss

def train(max_epochs, model_file, lr, train_encoder=True, train_decoder=True,
          train_transition=False):

    parameters = []
    if not args.cswm:    
        if train_transition:
            parameters = chain(parameters, model.transition_parameters())
        if train_encoder:
            parameters = chain(parameters, model.encoder_parameters())
        if train_decoder:
            parameters = chain(parameters, model.decoder_parameters())

        optimizer = torch.optim.Adam(parameters, lr = lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr = lr)

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
            obs, action, next_obs = data_batch

            optimizer.zero_grad()

            
            loss = 0.0

            if not args.cswm:
                state, kl_loss = model.encode(obs)
                if train_encoder or train_decoder:
                    rec_state = torch.sigmoid(model.decoder(state))
                    loss += (F.binary_cross_entropy(
                        rec_state, obs, reduction='sum') + kl_loss)

                if train_transition:
                    next_state, _ = model.encode(next_obs)

                    transition_loss, _, pred_next = model.transition(
                        state, action, next_state)
                    loss += transition_loss

                    if train_encoder and train_decoder:
                        loss += F.binary_cross_entropy(torch.sigmoid(model.decoder(pred_next)), next_obs,
                                        reduction='sum')
            else:
                loss = model.contrastive_loss(*data_batch)

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

def reload_model(model, filename):
    #only reloading encoder
    encoder_state = model.encoder.state_dict()
    decoder_state = model.decoder.state_dict()
    model_dict = model.state_dict()
    reload_dict = torch.load(filename)

    for name, param in reload_dict.items():
        if name.startswith("encoder.") or name.startswith("decoder."):
            if name not in model_dict:
                import pdb; pdb.set_trace()
            else:
                model_dict[name].copy_(param)
        else:
            if not name.startswith("transition_nets."):
                print(name)


if args.reload:
    if os.path.isfile(reload_file):
        reload_model(model, reload_file)
        evaluate(model_file, train_loader, False)
    else:
        print (str(reload_file) + "File not exist")

if not args.cswm:
    train(args.pretrain_epochs, model_file, lr=args.lr, train_encoder=True, train_transition=False, train_decoder=True)
    train(args.epochs, model_file, lr=args.transit_lr, train_encoder=False, train_transition=True, train_decoder=False)
    train(args.epochs, finetune_file, lr=args.lr, train_encoder=True, train_transition=True, train_decoder=True)
else:
    train(args.epochs, model_file, lr = args.lr)

if args.eval_dataset is not None:
    utils.eval_steps(
        model, [1, 5, 10],
        filename=args.eval_dataset, batch_size=args.batch_size,
        save_folder = save_folder, device=device, action_dim = args.action_dim, cswm  = args.cswm)