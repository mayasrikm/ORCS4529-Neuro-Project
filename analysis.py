import numpy as np
import neurogym as ngym

from tasks import PerceptualDecisionMaking
import matplotlib.pyplot as plt


class Analytic():

    def __init__(self, env : ngym.TrialEnv, obs_size) -> None:

        #!TODO: update agenty info based on how the code is built

        self.env   = env
        self.n     = obs_size

        self.activations = np.zeros((self.n, 1))
        for i in range(len(self.activations)):
            self.activations[i] = i

    def add_actvs(self, actvs : list):
        """
        Input: list of activations from evaluating state of environment
                hstack new list of activations to the end of self.activations
        """

        self.activations = np.hstack((self.activations, actvs))

    def get_canon_order(self) -> list:
        """
        Output: returns the order of numbers that make up the canonical numbering list
                    take the mean activations of each row
        """
        means = np.zeros((self.n, 2))
        for i in range(self.n):
            means[i][0] = i
            means[i][1] = np.mean(self.activations[i, 1:])
        c_ord = means[means[:, 1].argsort()][:, 0]
        return c_ord

    def get_canon_actvs(self, c_ord : list):
        """
        Output: returns a matrix where the rows are in canonical order
                    based on means of each row
        """
        canon_actvs = []
        for i in range(self.n):
            canon_actvs.append(self.activations[int(c_ord[i]), 1:])
        return canon_actvs

# ANALYSIS LOOP
"""
    1) Collect states + respective activations in order of a certain feature progression
        a) sort states by ground truth in some order to get the feature progression
    2) For each feature progression, create an Analytic object
        a) get canonical order of 1st feature and use it for all the later ones
    3) Plot!
"""
env = PerceptualDecisionMaking(dt=20)

def eval(env, trial_num = 100):
    _ = env.reset()
    states    = {} # dict of gt : state

    # Collect the states
    for _ in range(trial_num):

        trial = env.new_trial()
        ob, gt = env.ob, env.gt

        states[gt] = ob

        action = env.action_space.sample()  # A random agent #!TODO: update based on the different agents
        ob, reward, done, info = env.step(action)

    # Sort the states by ground truth
    gt_list = list(states.keys())
    gt_list.sort()

    pdm_analy = Analytic(env)

    # Input states to the Analytic object
    for i in range(len(gt_list)):
        s = states[gt_list[i]]

        V = w @ s.T                  # (num_neurons, num_objects) list of values for each neuron
        D =  B @ V

        pdm_analy.add_actvs(D)

    pdm_c_ord = pdm_analy.get_canon_order()
    pdm_actvs = pdm_analy.get_canon_actvs(pdm_c_ord)

    # Plot!
    plt.imshow(pdm_actvs, cmap='hot', interpolation='nearest')
    plt.show()
