import argparse
import torch
import torch.nn.functional as F
import pickle
from pathlib import Path

from torch.utils import data
import numpy as np
import tqdm

from cswm import utils
from cswm.models.modules import RewardPredictor, CausalTransitionModel, ContrastiveSWM
from cswm.utils import OneHot

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
args_eval = parser.parse_args()


meta_file = args_eval.save_folder / 'metadata.pkl'
if args_eval.finetune:
    model_file = args_eval.save_folder / 'finetuned_model.pt'
elif args_eval.random:
    model_file = args_eval.save_folder / 'random_model.pt'
else:
    model_file = args_eval.save_folder / 'model.pt'

if args_eval.finetune:
    reward_model_file = args_eval.save_folder / 'finetuned_reward_model.pt'
elif args_eval.random:
    reward_model_file = args_eval.save_folder / 'random_reward_model.pt'
else:
    reward_model_file = args_eval.save_folder / 'reward_model.pt'

with open(meta_file, 'rb') as f:
    args = pickle.load(f)['args']

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

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

print("VAE: ", args.vae)
print("Modular: ", args.modular)
print("Learn Edges: ", args.learn_edges)
print("Encoder: ", args.encoder)
print("Num Objects: ", args.num_objects)
print("Dataset: ", args.dataset)

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

if not args_eval.random:
    model.load_state_dict(torch.load(model_file))
else:
    torch.save(model.state_dict(), model_file)

model.eval()

Reward_Model = RewardPredictor(args.embedding_dim * args.num_objects).to(device)

def evaluate(valid_loader):
    valid_loss = 0.0
    Reward_Model.eval()

    for batch_idx, data_batch in enumerate(valid_loader):
        data_batch = [tensor.to(device).float() for tensor in data_batch]
        obs, _, _, reward, target = data_batch

        state, _ = model.encode(obs)
        reward_state, _ = model.encode(target)

        state_emb = torch.cat([state, reward_state], dim=1)
        reward_pred = Reward_Model(state_emb).view(reward.shape)

        loss = F.mse_loss(reward_pred, reward)

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
            obs, _, _, reward, target = data_batch

            optimizer.zero_grad()

            state, _ = model.encode(obs)
            reward_state, _ = model.encode(target)

            state_emb = torch.cat([state, reward_state], dim=1)
            reward_pred = Reward_Model(state_emb).view(reward.shape)

            loss = F.mse_loss(reward_pred, reward)

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

train(args.epochs, lr = args.lr)