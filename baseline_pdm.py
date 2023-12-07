import numpy as np
import torch
import torch.nn as nn

import neurogym as ngym

from tasks import AverageDirectionTest
from tasks import PerceptualDecisionMaking

import gym

from analysis import Analytic

import matplotlib.pyplot as plt

from re import template
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import neurogym as ngym
from tqdm import tqdm

env = PerceptualDecisionMaking()
_ = env.reset()
ob_size = np.shape(env.ob)[0]*np.shape(env.ob)[1]
act_size = env.action_space.n
num_neurons = ob_size
episodes = 2000
epsilon = 0.1


#Initialize variables
alpha = 0.1
gamma = 0.9
training_episodes = 20000
phi = np.zeros((1, ob_size))          # (1, obs_size) denotes inputs from environment, zeros are placeholder for what the actual input values are
w = np.random.randn(ob_size)  # (obs_size, 1) weights used to learn importance of each input

Q = np.zeros((ob_size, act_size))

for i in tqdm(range(episodes)):
    
    trial = env.new_trial()
    phi, gt = env.ob.flatten(), env.gt
    if np.max(Q) == 0 or np.random.rand() < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q) % act_size
    V = np.multiply(phi,w) 
    Q[:, action] = Q[:, action]+alpha*V
        
    phi_new, r, done, _ = env.step(action)
    rpe = r+gamma*(np.multiply(phi, w)-V) 
    w = w + alpha*phi*rpe  


def eval(env, obs_size, trial_num = 100):
    _ = env.reset()
    pdm_analy = Analytic(env, obs_size)


    # Collect the states
    for targ in range(act_size):

        for _ in range(trial_num):        

            trial = env.new_trial()
            ob, gt = env.ob, env.gt
            

            phi, gt = env.ob.flatten(), env.gt
            if np.max(Q) == 0 or np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q) % act_size

            V = np.multiply(phi, w)        # (num_neurons, num_objects) list of values for each neuron
            D = V

            if trial['ground_truth'] == targ:
                pdm_analy.add_actvs(np.array([D]).T)


            action = env.action_space.sample()  # A random agent #!TODO: update based on the different agents
            ob, reward, done, info = env.step(action)

        if targ == 0:
            pdm_c_ord = pdm_analy.get_canon_order()
        pdm_actvs = pdm_analy.get_canon_actvs(pdm_c_ord)

    # Plot!
    plt.imshow(pdm_actvs, cmap='RdYlGn', interpolation='nearest')
    plt.title(f"Baseline Perceptual Decision Making Heat Map of Activations")
    plt.xlabel("Trial Number")
    plt.ylabel("Neuron Number")
    plt.show()

eval(env, ob_size)
