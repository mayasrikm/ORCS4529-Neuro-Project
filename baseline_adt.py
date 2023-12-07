import numpy as np
import torch
import torch.nn as nn

import neurogym as ngym

from tasks import AverageDirectionTest

from analysis import Analytic

import matplotlib.pyplot as plt


# Environment
env = AverageDirectionTest()
_ = env.reset()
ob_size = env.observation_space.shape[0]
act_size = env.action_space.n
num_neurons = ob_size

#Initialize variables
alpha = 0.1
gamma = 0.9
training_episodes = 2000
phi = np.zeros((1, ob_size))          # (1, obs_size) denotes inputs from environment, zeros are placeholder for what the actual input values are
w = np.random.randn(ob_size)  # (obs_size, 1) weights used to learn importance of each input
#create sparse array B
B = np.zeros((ob_size, ob_size)) #(num_neurons, num_inputs) represents weighted combination of what input values each neuron sees
sparsity = 1
indices = np.random.choice(ob_size*ob_size, round(ob_size*ob_size*sparsity), replace=False)
B.flat[indices] = np.random.randn(round(ob_size*ob_size*sparsity))
B_inv = np.linalg.inv(B)

Q = np.zeros((ob_size, act_size))

epsilon = 0.1

#Train loop -- need to figure out how to get scalar V for each neuron
for i in range(training_episodes): 

    trial = env.new_trial()
    done = False

    while not done:
        phi, gt = env.space_obs, env.gt
        if np.max(Q) == 0 or np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q) % act_size

        V = np.multiply(phi, w)        # (num_neurons, num_objects) list of values for each neuron

        # take action based on most extreme value
        Q[:, action] = Q[:, action] + alpha*V
        phi_new, reward, done, misc_dict = env.step(action)
        phi = phi_new.flatten()
        # r = env.get_reward_vals         # (num_neurons, 1) correct decisions -- get_reward_vals needs to return the correct decision given the environment state
        rpe = reward+gamma*(np.multiply(phi, w)-V) 
        w = w + alpha*phi*rpe  

def eval(env, obs_size, trial_num = 100):
    _ = env.reset()
    pdm_analy = Analytic(env, obs_size)


    # Collect the states
    for targ in range(act_size):

        for _ in range(trial_num): 

            done = False

            while not done:       

                trial = env.new_trial()
                phi, gt = env.space_obs, env.gt

                if np.max(Q) == 0 or np.random.rand() < epsilon:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(Q) % act_size

                V = np.multiply(phi, w)        # (num_neurons, num_objects) list of values for each neuron
                D = V                      # (num_neurons, 1) list of decisions between left (0) and right (1)

                if gt == targ:
                    pdm_analy.add_actvs(np.array([D]).T)

                ob, reward, done, info = env.step(action)

        if targ == 0:
            pdm_c_ord = pdm_analy.get_canon_order()
        pdm_actvs = pdm_analy.get_canon_actvs(pdm_c_ord)

        print(len(pdm_analy.activations[0]))

    # Plot!
    plt.imshow(pdm_actvs)
    plt.title("Baseline Average Direction Test Heat Map of Activations")
    plt.show()

eval(env, ob_size)
