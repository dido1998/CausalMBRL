import argparse
from pathlib import Path
import pickle
import numpy as np
from tqdm import tqdm
import envs
import gym
import torch

from cswm.models.modules import CausalTransitionModel, ContrastiveSWM, RewardPredictor
from cswm import utils
from cswm.utils import OneHot
from torch.utils import data

def get_best_action(env):
    n = env.action_space.n
    best_reward = -np.inf
    best_action = None

    for i in range(n):
        reward, _ = env.unwrapped.sample_step(i)
        if reward > best_reward:
            best_reward = reward
            best_action = i

    return best_action

def get_best_model_action(obs, target, action_space, model, reward_model):
    n = env.action_space.n
    best_reward = -np.inf
    best_action = None
    target = torch.tensor(target).cuda().float().unsqueeze(0)

    for i in range(n):
        _, obs = env.unwrapped.sample_step(i)
        obs = torch.tensor(obs).cuda().float().unsqueeze(0)

        state, _ = model.encode(obs)
        target_state, _ = model.encode(target)

        emb = torch.cat([state, target_state], dim=1)

        reward_pred = reward_model(emb).detach().cpu().item()

        if reward_pred > best_reward:
            best_reward = reward_pred
            best_action = i

    return best_action

def planning_model(env, model, reward_model, episode_count, num_steps=20):
    action_space = env.action_space
    rewards = []

    for _ in tqdm(range(episode_count), leave=False):
        obs, target = env.reset()

        for i in range(num_steps):
            action = get_best_model_action(obs[1], target[1], 
                action_space, model, reward_model)

            obs, reward, _, _ = env.step(action)

        rewards.append(reward)

    rewards = np.array(rewards)
    success = rewards == 0.0

    print("Mean: ", np.mean(rewards))
    print("Standard Deviation: ", np.std(rewards))
    print("Success Rate: ", np.mean(success))

def planning_best(env, episode_count, num_steps=20):
    action_space = env.action_space
    rewards = []

    for _ in tqdm(range(episode_count), leave=False):
        _, _ = env.reset()

        for i in range(num_steps):
            action = get_best_action(env)
            _, reward, _, _ = env.step(action)

        rewards.append(reward)

    rewards = np.array(rewards)
    success = rewards == 0.0

    print("Mean: ", np.mean(rewards))
    print("Standard Deviation: ", np.std(rewards))
    print("Success Rate: ", np.mean(success))

def planning_random(env, episode_count, num_steps=20):
    action_space = env.action_space
    rewards = []

    for _ in tqdm(range(episode_count), leave=False):
        _, _ = env.reset()

        for i in range(num_steps):
            action = action_space.sample()
            _, reward, _, _ = env.step(action)

        rewards.append(reward)

    rewards = np.array(rewards)
    success = rewards == 0.0

    print("Mean: ", np.mean(rewards))
    print("Standard Deviation: ", np.std(rewards))
    print("Success Rate: ", np.mean(success))


parser = argparse.ArgumentParser()
parser.add_argument('--save-folder', type=Path,
                    default='checkpoints',
                    help='Path to checkpoints.')
parser.add_argument('--finetune', action='store_true', default=False,
                    help='Whether to use finetuned model')
parser.add_argument('--random', action='store_true', default=False,
                    help='Whether to use finetuned model')

args_eval = parser.parse_args()

num_eval = 1000
meta_file = args_eval.save_folder / 'metadata.pkl'

finetune_model_file = args_eval.save_folder / 'finetuned_model.pt'
finetune_reward_model_file = args_eval.save_folder / 'finetuned_reward_model.pt'

random_model_file = args_eval.save_folder / 'random_model.pt'
random_reward_model_file = args_eval.save_folder / 'random_reward_model.pt'

model_file = args_eval.save_folder / 'model.pt'
reward_model_file = args_eval.save_folder / 'reward_model.pt'

with open(meta_file, 'rb') as f:
    args = pickle.load(f)['args']

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

print("Loading model...")

input_shape = [3,50,50]

print(input_shape)
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
        encoder=args.encoder).cuda()
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
        multiplier=args.multiplier).cuda()

    num_enc = sum(p.numel() for p in model.encoder_parameters())
    num_dec = sum(p.numel() for p in model.decoder_parameters())
    num_tr = sum(p.numel() for p in model.transition_parameters())

Reward_Model = RewardPredictor(args.embedding_dim * args.num_objects).cuda()

with gym.make('WShapesRL-Observed-Train-3-Blues-v0') as env:
    print("Random Planning: ")
    planning_random(env, num_eval)
    print()

    print("Best Planning: ")
    planning_best(env, num_eval)
    print()

    if args_eval.random:
        model.load_state_dict(torch.load(random_model_file))
        Reward_Model.load_state_dict(torch.load(random_reward_model_file))

        Reward_Model.eval()
        model.eval()

        print("Random Model Planning: ")
        planning_model(env, model, Reward_Model, num_eval)
        print()

    if args_eval.finetune:
        model.load_state_dict(torch.load(finetune_model_file))
        Reward_Model.load_state_dict(torch.load(finetune_reward_model_file))

        Reward_Model.eval()
        model.eval()

        print("Finetuned Model Planning: ")
        planning_model(env, model, Reward_Model, num_eval)
        print()

    model.load_state_dict(torch.load(model_file))
    Reward_Model.load_state_dict(torch.load(reward_model_file))

    Reward_Model.eval()
    model.eval()

    print("Model Planning: ")
    planning_model(env, model, Reward_Model, num_eval)
    print()
