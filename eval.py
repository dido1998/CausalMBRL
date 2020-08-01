 
import argparse
import torch
import pickle
from pathlib import Path

from torch.utils import data
import numpy as np

from cswm import utils
from cswm.models.modules import CausalTransitionModel, ContrastiveSWM, ContrastiveSWMFinal
from cswm.utils import OneHot

torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('--save-folder', type=Path,
                    default='checkpoints',
                    help='Path to checkpoints.')

parser.add_argument('--dataset', type=Path,
                    default=Path('data/shapes_eval.h5'),
                    help='Dataset file name.')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disable CUDA training.')

args_eval = parser.parse_args()


meta_file = args_eval.save_folder / 'metadata.pkl'
model_file = args_eval.save_folder / 'model.pt'

with open(meta_file, 'rb') as f:
    args = pickle.load(f)['args']

args.cuda = not args_eval.no_cuda and torch.cuda.is_available()
args.batch_size = 100
args.dataset = args_eval.dataset
args.seed = 0

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

device = torch.device('cuda' if args.cuda else 'cpu')

print("Loading data...")
dataset = utils.PathDataset(
    hdf5_file=args.dataset, path_length=10,
    action_transform=OneHot(args.num_objects * args.action_dim), in_memory=False)
eval_loader = data.DataLoader(
    dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

print("Loading model...")

# Get data sample
obs = next(iter(eval_loader))[0]
input_shape = obs[0][0].size()

if args.cswm:
    model = ContrastiveSWMFinal(
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

model.load_state_dict(torch.load(model_file))
model.eval()

utils.eval_steps(
    model, [1, 5, 10],
    filename=args.dataset, batch_size=args.batch_size,
    device=device, save_folder='Experiments', cswm=args.cswm)
