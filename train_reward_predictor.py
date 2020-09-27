import argparse
import torch
import torch.nn.functional as F
import pickle
from pathlib import Path

from torch.utils import data
import numpy as np
import tqdm
import logging

import random
import torch
from cswm import utils
from cswm.models.modules import RewardPredictor, CausalTransitionModel
from cswm.utils import OneHot

import sys
import datetime
import os
import pickle
from pathlib import Path

import numpy as np

from itertools import chain

torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('--save-folder', type=Path,
                    default='checkpoints',
                    help='Path to checkpoints.')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disable CUDA training.')
parser.add_argument('--finetune', action='store_true', default=False,
                    help='Whether to use finetuned model')
parser.add_argument('--random', action='store_true', default=False)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--recurrent', action = 'store_true')
args_eval = parser.parse_args()

def set_seed(seed):
    """Set seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.cuda:
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True

meta_file = args_eval.save_folder / 'metadata.pkl'
if args_eval.finetune:
    model_file = args_eval.save_folder / 'finetuned_model.pt'
    reward_model_file = args_eval.save_folder / 'finetune_reward_model.pt'
    log_file = args_eval.save_folder / 'finetune_reward_log.txt' 
elif args_eval.random:
    model_file = args_eval.save_folder / 'random_model.pt'
    reward_model_file = args_eval.save_folder / 'random_reward_model.pt'
    log_file = args_eval.save_folder / 'random_reward_log.txt'
else:
    model_file = args_eval.save_folder / 'model.pt'
    reward_model_file = args_eval.save_folder / 'reward_model.pt'
    log_file = args_eval.save_folder / 'reward_log.txt'

with open(meta_file, 'rb') as f:
    args = pickle.load(f)['args']

handlers = [logging.FileHandler(log_file, 'a')]
if args.silent:
    handlers.append(logging.StreamHandler(sys.stdout))
logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=handlers)
logger = logging.getLogger()
print = logger.info

set_seed(args.seed)

device = torch.device('cuda' if args.cuda else 'cpu')

print("Loading data...")
dataset = utils.StateTransitionsDataset(
    hdf5_file=args.dataset, action_transform=OneHot(args.num_objects * args.action_dim))
valid_dataset = utils.StateTransitionsDataset(
    hdf5_file=args.valid_dataset, action_transform=OneHot(args.num_objects * args.action_dim))

train_loader = data.DataLoader(
    dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
valid_loader = data.DataLoader(
    valid_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

print("Loading model...")

# Get data sample
obs = next(iter(train_loader))[0]
input_shape = obs[0].size()

print(f"VAE: {args.vae}")
print(f"Modular: {args.modular}")
print(f"Encoder: {args.encoder}")
print(f"Num Objects: {args.num_objects}")
print(f"Dataset: {args.dataset}")

if not args_eval.recurrent:
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

else:
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

num_enc = sum(p.numel() for p in model.encoder_parameters())
num_dec = sum(p.numel() for p in model.decoder_parameters())
num_tr = sum(p.numel() for p in model.transition_parameters())

print(f'Number of parameters in Encoder: {num_enc}')
print(f'Number of parameters in Decoder: {num_dec}')
print(f'Number of parameters in Transition: {num_tr}')
print(f'Number of parameters: {num_enc+num_dec+num_tr}')

if not args_eval.random:
    model.load_state_dict(torch.load(model_file))
else:
    torch.save(model.state_dict(), model_file)

model.eval()

Reward_Model = RewardPredictor(args.embedding_dim_per_object * args.num_objects).to(device)

def evaluate(valid_loader):
    valid_loss = 0.0
    Reward_Model.eval()

    for batch_idx, data_batch in enumerate(valid_loader):
        data_batch = [tensor.to(device).float() for tensor in data_batch]
        _, _, obs, reward, target = data_batch

        state, _ = model.encode(obs)
        state = state.view(state.shape[0], args.num_objects * args.embedding_dim_per_object)
        reward_state, _ = model.encode(target)
        reward_state = reward_state.view(state.shape[0], args.num_objects * args.embedding_dim_per_object)

        state_emb = torch.cat([state, reward_state], dim=1)
        reward_pred = Reward_Model(state_emb).view(reward.shape)

        loss = F.l1_loss(reward_pred, reward)

        valid_loss += loss.item()
    
    avg_loss = valid_loss / len(valid_loader)
    print('====> Average valid loss: {:.6f}'.format(avg_loss))
    return avg_loss

def train(max_epochs, lr):

    optimizer = torch.optim.Adam(Reward_Model.parameters(), lr = lr)

    print('Starting model training...')
    best_loss = 1e9
    for epoch in range(1, max_epochs + 1):
        train_loss = 0

        iterator = tqdm.tqdm(train_loader, desc=f'Epoch {epoch}',
                             disable=args.silent)

        for batch_idx, data_batch in enumerate(iterator):
            Reward_Model.train()
            data_batch = [tensor.to(device).float() for tensor in data_batch]
            _, _, obs, reward, target = data_batch

            optimizer.zero_grad()
            #print(obs.size())
            state, _ = model.encode(obs)
            state = state.view(state.shape[0], args.num_objects * args.embedding_dim_per_object)
            #print(target.size())
            reward_state, _ = model.encode(target)
            reward_state = reward_state.view(state.shape[0], args.num_objects * args.embedding_dim_per_object)

            state_emb = torch.cat([state, reward_state], dim=1)
            reward_pred = Reward_Model(state_emb).view(reward.shape)

            loss = F.l1_loss(reward_pred, reward)

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

        avg_loss = evaluate(valid_loader)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(Reward_Model.state_dict(), reward_model_file)

train(args_eval.epochs, lr = args.lr)
