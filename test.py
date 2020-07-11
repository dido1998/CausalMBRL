import h5py
import gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import envs

env = gym.make('WShapesRL-Unobserved-Train-5-Sets-v0')
matplotlib.use('TkAgg')
plt.axis('off')
obs, target = env.reset()

obs = np.transpose(obs[1], [1,2,0])
target = np.transpose(target[1], [1,2,0])

print(obs.shape)
print(target.shape)

plt.subplot(221)
plt.imshow(obs)
plt.subplot(222)
plt.imshow(target)
plt.show()

for i in range(50):

    obj_idx = int(input())
    direction = int(input())
    obs, reward, _, _ = env.step(obj_idx * 4 + direction)
    obs = np.transpose(obs[1], [1,2,0])

    plt.subplot(221)
    plt.imshow(obs)
    plt.subplot(222)
    plt.imshow(target)
    plt.show()

    print(reward)