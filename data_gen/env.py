"""Simple random agent.
Running this script directly executes the random agent in environment and stores
experience in a replay buffer.
"""

# Get env directory
import sys
from tqdm import tqdm
from pathlib import Path
if str(Path.cwd()) not in sys.path:
    sys.path.insert(0, str(Path.cwd()))

import argparse

# noinspection PyUnresolvedReferences
import envs

from cswm import utils

import gym
from gym import logger

import numpy as np
from PIL import Image
import torch

class RandomAgent(object):
    """The world's simplest agent!"""

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        del observation, reward, done
        return self.action_space.sample()


def crop_normalize(img, crop_ratio):
    img = img[crop_ratio[0]:crop_ratio[1]]
    img = Image.fromarray(img).resize((50, 50), Image.ANTIALIAS)
    return np.transpose(np.array(img), (2, 0, 1)) / 255


def parse_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env-id', type=str, default='ShapesTrain-v0',
                        help='Select the environment to run.')
    parser.add_argument('--fname', type=str, default='data/shapes_train.h5',
                        help='Save path for replay buffer.')
    parser.add_argument('--num-episodes', type=int, default=1000,
                        help='Total number of episodes to simulate.')
    parser.add_argument('--atari', action='store_true', default=False,
                        help='Run atari mode (stack multiple frames).')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed.')
    parser.add_argument('--episode-length', type=int, default=100)
    parser.add_argument('--silent', action='store_true')
    parser.add_argument('--save_graph', action = 'store_true')
    parser.add_argument('--save_graph_location', type = str)
    parser.add_argument('--load_graph', action = 'store_true')
    parser.add_argument('--load_graph_location', type = str)
    parser.add_argument('--graph', type = str, default = None)
    return parser.parse_args()


def try_generate_episode(env, agent, *, warm_start, crop,
                         episode_length, atari):
    reward = 0
    done = False

    obs = []
    actions = []
    next_obs = []
    rewards = []
    goals = []

    ob, target = env.reset()

    if atari:
        # Burn-in steps
        for _ in range(warm_start):
            action = agent.act(ob, reward, done)
            ob, _, _, _ = env.step(action)
        prev_ob = crop_normalize(ob, crop)
        ob, _, _, _ = env.step(0)
        ob = crop_normalize(ob, crop)

        while True:
            obs.append(
                np.concatenate((ob, prev_ob), axis=0))
            prev_ob = ob

            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            ob = crop_normalize(ob, crop)

            actions.append(action)
            next_obs.append(
                np.concatenate((ob, prev_ob), axis=0))

            if done:
                break
    else:
        while len(obs) < episode_length:
            action = agent.act(ob, reward, done)
            next_ob, reward, done, info = env.step(action)

            if info is not None and info.get('invalid_push', False):
                done = False
                continue

            obs.append(ob[1])
            actions.append(action)
            next_obs.append(next_ob[1])
            rewards.append(reward)

            ob = next_ob

            if done:
                break

    if len(actions) < 10:
        return None

    return dict(
        obs=obs,
        action=actions,
        next_obs=next_obs,
        reward=rewards,
        target=target[1],
    )


def generate_episode(env, agent, **kwargs):
    episode = None
    while episode is None:
        episode = try_generate_episode(env, agent, **kwargs)
    return episode


def generate(env, args):
    np.random.seed(args.seed)
    env.action_space.seed(args.seed)
    env.seed(args.seed)

    agent = RandomAgent(env.action_space)

    episode_count = args.num_episodes

    crop = None
    warm_start = None
    if args.env_id == 'PongDeterministic-v4':
        crop = (35, 190)
        warm_start = 58
    elif args.env_id == 'SpaceInvadersDeterministic-v4':
        crop = (30, 200)
        warm_start = 50

    if args.atari:
        env._max_episode_steps = warm_start + 11

    replay_buffer = []

    for _ in tqdm(range(episode_count), leave=False, disable=args.silent):
        episode = generate_episode(
            env, agent, warm_start=warm_start, crop=crop,
            episode_length=args.episode_length, atari=args.atari)
        replay_buffer.append(episode)
    return replay_buffer


def main():
    args = parse_args()

    logger.set_level(logger.INFO)

    with gym.make(args.env_id) as env:
        if 'ColorChanging' in args.env_id:
            if args.load_graph:
                graph = torch.load(args.load_graph_location)
                env.unwrapped.load_save_information(graph)
            elif args.graph != 'None':
                env.unwrapped.set_graph(args.graph)
                
        replay_buffer = generate(env, args)
        if 'ColorChanging' in args.env_id and args.save_graph:
            graph = env.unwrapped.get_save_information()
            torch.save(graph, args.save_graph_location)

    # Save replay buffer to disk.
    utils.save_list_dict_h5py(replay_buffer, args.fname)
    print('Done.')


if __name__ == '__main__':
    main()
