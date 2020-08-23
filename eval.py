 
import argparse
import torch
import pickle
from pathlib import Path

from torch.utils import data
import numpy as np

from cswm import utils
from cswm.models.modules import CausalTransitionModel, CausalTransitionModelLSTM
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
parser.add_argument('--finetune', action='store_true')
parser.add_argument('--recurrent', action='store_true')
parser.add_argument('--save', type=str, default='Default')
args_eval = parser.parse_args()


meta_file = args_eval.save_folder / 'metadata.pkl'
if args_eval.finetune:
    model_file = args_eval.save_folder / 'finetuned_model.pt'
else:
    model_file = args_eval.save_folder / 'model.pt'

with open(meta_file, 'rb') as f:
    args = pickle.load(f)['args']

args.cuda = not args_eval.no_cuda and torch.cuda.is_available()
args_eval.batch_size = 100

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

model.load_state_dict(torch.load(model_file))
model.eval()

model_name = '/'.join(str(args_eval.save_folder).split('/')[-2:])

if args_eval.recurrent:
    utils.eval_steps_lstm(
        model, [1,5,10], name=model_name,
        filename=args_eval.dataset, batch_size=args_eval.batch_size,
        device=device, save_folder=args_eval.save, contrastive=args.contrastive)
else:
    utils.eval_steps(
        model, [1, 5, 10], name=model_name,
        filename=args_eval.dataset, batch_size=args_eval.batch_size,
        device=device, save_folder=args_eval.save, contrastive=args.contrastive)
