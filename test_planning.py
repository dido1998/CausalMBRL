import argparse
from pathlib import Path
import pickle
import numpy as np
from tqdm import tqdm
import envs
import gym
import torch
import torch.nn.functional as F

from cswm.models.modules import CausalTransitionModel, RewardPredictor

from cswm import utils
from cswm.utils import OneHot
from torch.utils import data

parser = argparse.ArgumentParser()
parser.add_argument('--save-folder', type=Path,
                    default='checkpoints',
                    help='Path to checkpoints.')
parser.add_argument('--finetune', action='store_true', default=False,
                    help='Whether to use finetuned model')
parser.add_argument('--random', action='store_true', default=False,
                    help='Whether to use finetuned model')
parser.add_argument('--env-id', type = str)
parser.add_argument('--num-eval', type=int, default=1000)
parser.add_argument('--num-steps', type=int, default=1)
parser.add_argument('--save', type=str, default='Experiments')
parser.add_argument('--recurrent', action='store_true')

args_eval = parser.parse_args()

string=""

num_steps = args_eval.num_steps
num_eval = args_eval.num_eval

def get_success(rewards):
    if 'ColorChanging' in args_eval.env_id:
        return rewards == 1.0
    else:
        return rewards == 0.0

def get_best_action(env):
    n = env.action_space.n
    best_reward = -np.inf
    best_action = None
    if 'ColorChanging' in args_eval.env_id:
        best_action = env.unwrapped.sample_step()
        return best_action
    else: 
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
        state = state.view(state.shape[0], args.num_objects * args.embedding_dim_per_object)
        target_state, _ = model.encode(target)
        target_state = target_state.view(target_state.shape[0], args.num_objects * args.embedding_dim_per_object)

        emb = torch.cat([state, target_state], dim=1)

        reward_pred = reward_model(emb).detach().cpu().item()

        if reward_pred > best_reward:
            best_reward = reward_pred
            best_action = i

    return best_action

def get_best_model_action_transition(state, target_state, action_space, model, reward_model):
    n = env.action_space.n
    best_reward = -np.inf
    best_action = None
    state_out = None

    for i in range(n):
        action = F.one_hot(torch.tensor(i).long(), num_classes=n).cuda().float().unsqueeze(0)
        next_state = model.transition(state, action)

        emb = torch.cat([next_state.view(next_state.shape[0], args.num_objects * args.embedding_dim_per_object),
                         target_state.view(target_state.shape[0], args.num_objects * args.embedding_dim_per_object)],
                         dim = 1)

        reward_pred = reward_model(emb).detach().cpu().item()

        if reward_pred > best_reward:
            best_reward = reward_pred
            best_action = i 
            state_out = next_state

    return best_action, state_out

def planning_model(env, model, reward_model, episode_count):
    global string
    action_space = env.action_space
    rewards = []

    for _ in tqdm(range(episode_count), leave=False):
        obs, target = env.reset(num_steps=num_steps)

        for i in range(num_steps):
            action = get_best_model_action(obs[1], target[1], 
                action_space, model, reward_model)

            obs, reward, _, _ = env.step(action)

        rewards.append(reward)

    rewards = np.array(rewards)
    success = get_success(rewards)

    mean_rewards = str(np.around(np.mean(rewards), decimals=2))
    std_rewards = str(np.around(np.std(rewards), decimals=2))
    mean_success = str(np.around(np.mean(success), decimals=2))

    print("Mean: ", mean_rewards)
    print("Standard Deviation: ", std_rewards)
    print("Success Rate: ", mean_success)

    string += mean_rewards + ' | ' + std_rewards + ' | ' + mean_success + ' | '

def planning_model_transition(env, model, reward_model, episode_count):
    global string
    action_space = env.action_space
    rewards = []

    for _ in tqdm(range(episode_count), leave=False):
        obs, target = env.reset(num_steps=num_steps)
        obs = torch.tensor(obs[1]).cuda().float().unsqueeze(0)
        target = torch.tensor(target[1]).cuda().float().unsqueeze(0)

        obs_state, _ = model.encode(obs)
        target_state, _ = model.encode(target)

        for i in range(num_steps):
            action, obs_state = get_best_model_action_transition(obs_state, target_state,
                action_space, model, reward_model)

            _, reward, _, _ = env.step(action)

        rewards.append(reward)

    rewards = np.array(rewards)
    success = get_success(rewards)

    mean_rewards = str(np.around(np.mean(rewards), decimals=2))
    std_rewards = str(np.around(np.std(rewards), decimals=2))
    mean_success = str(np.around(np.mean(success), decimals=2))

    print("Mean: ", mean_rewards)
    print("Standard Deviation: ", std_rewards)
    print("Success Rate: ", mean_success)

    string += mean_rewards + ' | ' + std_rewards + ' | ' + mean_success + ' | '

def planning_best(env, episode_count):
    action_space = env.action_space
    rewards = []

    for _ in tqdm(range(episode_count), leave=False):
        _, _ = env.reset(num_steps=num_steps)

        for i in range(num_steps):
            action = get_best_action(env)
            _, reward, _, _ = env.step(action)

        rewards.append(reward)

    rewards = np.array(rewards)
    success = get_success(rewards)

    print("Mean: ", np.mean(rewards))
    print("Standard Deviation: ", np.std(rewards))
    print("Success Rate: ", np.mean(success))

def planning_random(env, episode_count):
    action_space = env.action_space
    rewards = []

    for _ in tqdm(range(episode_count), leave=False):
        _, _ = env.reset(num_steps=num_steps)

        for i in range(num_steps):
            action = action_space.sample()
            _, reward, _, _ = env.step(action)

        rewards.append(reward)

    rewards = np.array(rewards)
    success = get_success(rewards)

    print("Mean: ", np.mean(rewards))
    print("Standard Deviation: ", np.std(rewards))
    print("Success Rate: ", np.mean(success))

#with gym.make(args_eval.env_id) as env:
#    #if 'ColorChanging' in args_eval.env_id:
#    #    env.unwrapped.load_save_information(torch.load(graph_location))
#    #print("Random Planning: ")
#    #planning_random(env, num_eval)
#    #print()

#    print("Best Planning: ")
#    planning_best(env, num_eval)
#    print()
#exit()

if 'ColorChanging' in args_eval.env_id:
    graph_location = 'data/ColorChangingRL'

meta_file = args_eval.save_folder / 'metadata.pkl'

finetune_model_file = args_eval.save_folder / 'finetune_model.pt'
finetune_reward_model_file = args_eval.save_folder / 'finetune_reward_model.pt'

random_model_file = args_eval.save_folder / 'random_model.pt'
random_reward_model_file = args_eval.save_folder / 'random_reward_model.pt'

model_file = args_eval.save_folder / 'model.pt'
reward_model_file = args_eval.save_folder / 'reward_model.pt'

with open(meta_file, 'rb') as f:
    args = pickle.load(f)['args']

if 'ColorChanging' in args_eval.env_id:
    graph_location = 'data/ColorChangingRL_'+str(args.num_objects)+'-'+str(args.action_dim)+'-'+args_eval.env_id.split('-')[-2]+'-Static-train-graph-'+str(args.dataset).split('.')[0].split('-')[-1]
    print('graph location:' + str(graph_location))

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

print("Loading model...")

input_shape = [3,50,50]

print(input_shape)
print("VAE: ", args.vae)
print("Modular: ", args.modular)
print("Encoder: ", args.encoder)
print("Num Objects: ", args.num_objects)
print("Dataset: ", args.dataset)

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
        copy_action=args.copy_action).cuda()

num_enc = sum(p.numel() for p in model.encoder_parameters())
num_dec = sum(p.numel() for p in model.decoder_parameters())
num_tr = sum(p.numel() for p in model.transition_parameters())

Reward_Model = RewardPredictor(args.embedding_dim_per_object * args.num_objects).cuda()

model_name = '/'.join(str(args_eval.save_folder).split('/')[-2:])

string += model_name + ' : '

with gym.make(args_eval.env_id) as env:
    if 'ColorChanging' in args_eval.env_id:
        env.unwrapped.load_save_information(torch.load(graph_location))
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
        planning_model_transition(env, model, Reward_Model, num_eval)
        print()

    if args_eval.finetune:
        model.load_state_dict(torch.load(finetune_model_file))
        Reward_Model.load_state_dict(torch.load(finetune_reward_model_file))

        Reward_Model.eval()
        model.eval()

        print("Finetuned Model Planning: ")
        planning_model(env, model, Reward_Model, num_eval)
        planning_model_transition(env, model, Reward_Model, num_eval)
        print()

    model.load_state_dict(torch.load(model_file))
    Reward_Model.load_state_dict(torch.load(reward_model_file))

    Reward_Model.eval()
    model.eval()

    print("Model Planning: ")
    planning_model(env, model, Reward_Model, num_eval)
    planning_model_transition(env, model, Reward_Model, num_eval)
    print()

string+='\n'
with open(args_eval.save+'/eval_rl_'+str(num_steps)+'.txt', 'a') as f:
    f.write(string)
