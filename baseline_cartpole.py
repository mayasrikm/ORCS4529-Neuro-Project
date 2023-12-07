import numpy as np
import torch
import torch.nn as nn

import neurogym as ngym
import gym
import tqdm

from tasks import DawTwoStep

from analysis import Analytic

import matplotlib.pyplot as plt


# Environment
envname = "CartPole-v0"
env = gym.make(envname)
ob_size = 64
act_size = env.action_space.n
num_neurons = ob_size

#Initialize variables
alpha = 0.1
gamma = 0.9
training_episodes = 200
sparsity = 0.8

phi = np.zeros((4, ob_size))          # (1, obs_size) denotes inputs from environment, zeros are placeholder for what the actual input values are
indices = np.random.choice(4*ob_size, round(4*ob_size*sparsity), replace=False)
phi.flat[indices] = np.random.randn(round(4*ob_size*sparsity))

w = np.random.randn(ob_size)  # (obs_size, 1) weights used to learn importance of each input

#create sparse array B
B = np.zeros((ob_size, ob_size)) #(num_neurons, num_inputs) represents weighted combination of what input values each neuron sees
indices = np.random.choice(ob_size*ob_size, round(ob_size*ob_size*sparsity), replace=False)
B.flat[indices] = np.random.randn(round(ob_size*ob_size*sparsity))
B_inv = np.linalg.inv(B)

Q = np.zeros((ob_size, act_size))

epsilon = 0.1

#Train loop -- need to figure out how to get scalar V for each neuron
for i in range(training_episodes): 

    obs, _ = env.reset()
    done = False

    while not done:
        s = obs @ phi
        if np.max(Q) == 0 or np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q) % act_size

        V = np.multiply(s, w)        # (num_neurons, num_objects) list of values for each neuron

        # take action based on most extreme value
        Q[:, action] = Q[:, action] + alpha*V
        obs, r, done, misc_dict, _ = env.step(action)
        rpe = r+gamma*(np.multiply(s, w)-V) 
        w = w + alpha*s*rpe  
    


def eval(env, obs_size, trial_num = 100):
    _ = env.reset()
    pdm_analy = Analytic(env, obs_size)

    feature_names = ['Cart Position', 'Cart Velocity', 'Pole Angle', 'Pole Angular Velocity']

    # Collect the states
    for i in range(4):

        feat = feature_names[i]

        state_collection = {}

        for _ in range(trial_num):        

            obs, _ = env.reset()
            done = False

            while not done:
                s = obs @ phi
                if np.random.rand() < 0.1:
                    state_collection[obs[i]] = s              
                if np.max(Q) == 0 or np.random.rand() < epsilon:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(Q) % act_size

                V = np.multiply(s, w)        # (num_neurons, num_objects) list of values for each neuron
                D = V                      # (num_neurons, 1) list of decisions between left (0) and right (1)
                obs, reward, done, info, _ = env.step(action)

        state_collection = dict(sorted(state_collection.items()))
        for k in state_collection.keys():
            s = state_collection[k]         

            V = np.multiply(s, w)        # (num_neurons, num_objects) list of values for each neuron
            D = V                      # (num_neurons, 1) list of decisions between left (0) and right (1)

            pdm_analy.add_actvs(np.array([D]).T)

        if i == 0:
            pdm_c_ord = pdm_analy.get_canon_order()
        pdm_actvs = pdm_analy.get_canon_actvs(pdm_c_ord)

    # Plot
    plt.imshow(pdm_actvs)
    plt.title("Baseline Cartpole Heat Map")
    plt.show()

eval(env, ob_size)
