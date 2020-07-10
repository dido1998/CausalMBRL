import h5py
import gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import envs

env = gym.make('WShapesUnobserved-Train-3-Sets-v0')
obs = env.reset()
matplotlib.use('TkAgg')
plt.imshow(np.transpose(obs[1], (1,2,0)))
plt.show()

for i in range(50):
    obj_idx = int(input())
    direction = int(input())
    obs, _, _, _ = env.step(obj_idx * 4 + direction)
    img = np.transpose(obs[1], [1,2,0])
    plt.axis('off')
    plt.imshow(img)
    plt.show()
    plt.close()
