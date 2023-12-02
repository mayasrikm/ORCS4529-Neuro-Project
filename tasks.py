import warnings
warnings.filterwarnings("ignore")  # to suppress warnings

import numpy as np
import neurogym as ngym

class PerceptualDecisionMaking(ngym.TrialEnv):
    """Two-alternative forced choice task in which the subject has to
    integrate two stimuli to decide which one is higher on average.

    Args:
        stim_scale: Controls the difficulty of the experiment. (def: 1., float)
        sigma: float, input noise level
        dim_ring: int, dimension of ring input and output
    """
    metadata = {
        'paper_link': 'https://www.jneurosci.org/content/12/12/4745',
        'paper_name': '''The analysis of visual motion: a comparison of
        neuronal and psychophysical performance''',
        'tags': ['perceptual', 'two-alternative', 'supervised']
    }

    def __init__(self, dt=100, rewards=None, timing=None, stim_scale=1.,
                 sigma=1.0, dim_ring=2):
        super().__init__(dt=dt)
        # The strength of evidence, modulated by stim_scale
        self.cohs = np.array([0, 6.4, 12.8, 25.6, 51.2]) * stim_scale
        self.sigma = sigma / np.sqrt(self.dt)  # Input noise

        # Rewards
        self.rewards = {'abort': -0.1, 'correct': +1., 'fail': 0.}
        if rewards:
            self.rewards.update(rewards)

        self.timing = {
            'fixation': 100,
            'stimulus': 2000,
            'delay': 0,
            'decision': 100}
        if timing:
            self.timing.update(timing)

        self.abort = False

        self.theta = np.linspace(0, 2*np.pi, dim_ring+1)[:-1]
        self.choices = np.arange(dim_ring)

        name = {'fixation': 0, 'stimulus': range(1, dim_ring+1)}
        self.observation_space = ngym.spaces.Box(
            -np.inf, np.inf, shape=(1+dim_ring,), dtype=np.float32, name=name)
        name = {'fixation': 0, 'choice': range(1, dim_ring+1)}
        self.action_space = ngym.spaces.Discrete(1+dim_ring, name=name)

    def _new_trial(self, **kwargs):
        # Trial info
        trial = {
            'ground_truth': self.rng.choice(self.choices),
            'coh': self.rng.choice(self.cohs),
        }
        trial.update(kwargs)

        coh = trial['coh']
        ground_truth = trial['ground_truth']
        stim_theta = self.theta[ground_truth]

        # Periods
        self.add_period(['fixation', 'stimulus', 'delay', 'decision'])

        # Observations
        self.add_ob(1, period=['fixation', 'stimulus', 'delay'], where='fixation')
        stim = np.cos(self.theta - stim_theta) * (coh/200) + 0.5
        self.add_ob(stim, 'stimulus', where='stimulus')
        self.add_randn(0, self.sigma, 'stimulus', where='stimulus')

        # Ground truth
        self.set_groundtruth(ground_truth, period='decision', where='choice')

        return trial

    def _step(self, action):
        new_trial = False
        # rewards
        reward = 0
        gt = self.gt_now
        # observations
        if self.in_period('fixation'):
            if action != 0:  # action = 0 means fixating
                new_trial = self.abort
                reward += self.rewards['abort']
        elif self.in_period('decision'):
            if action != 0:
                new_trial = True
                if action == gt:
                    reward += self.rewards['correct']
                    self.performance = 1
                else:
                    reward += self.rewards['fail']

        return self.ob_now, reward, False, {'new_trial': new_trial, 'gt': gt}
    
    
class AverageDirectionTest(ngym.TrialEnv):
    """
    On each trial, agent is given a noisy imput of K balls moving in left or right
    Agent must learn to select a region to take input information from and decide on whether all the balls are moving on net left or right.
    In each new state each ball in a given region has velocity gets *-1 if incorrect --> trajectory matching
    In next state each ball may be in a new region. Goal is to maximize reward.

    Args:
        
    """
    def __init__(self, dt=100, rewards=None, sigma=1,
                    K = 64, max_step = 10):
        super().__init__(dt=dt)
        # Possible decisions at the end of the trial
        self.choices = range(-10, 10)  # pseudo-continuous
        self.sigma   = sigma / np.sqrt(self.dt)  # Input noise
        self.K       = K

        # Optional rewards dictionary
        self.rewards = {'abort': -0.1, 'correct': +1., 'fail': 0.}
        if rewards:
            self.rewards.update(rewards)

        # Similar to gym envs, define observations_space and action_space
        # Optional annotation of the observation space
        self.observation_space = ngym.spaces.Box(
            -np.inf, np.inf, shape=(K), dtype=np.float32, name=name)
        
        # Optional annotation of the action space
        name = {'choice': self.choices}
        self.action_space = ngym.spaces.Discrete(2, name=name)

        self.total_steps = 0
        self.max_step = max_step


    def _new_trial(self, **kwargs):
        """
        self._new_trial() is called internally to generate a next trial.

        Typically, you need to
            set trial: a dictionary of trial information
            run self.add_period():
                will add time periods to the trial
                accesible through dict self.start_t and self.end_t
            run self.add_ob():
                will add observation to np array self.ob
            run self.set_groundtruth():
                will set groundtruth to np array self.gt

        Returns:
            trial: dictionary of trial information
        """

        self.total_steps = 0
        
        # Setting trial information
        speed_obs = np.random.rand(self.K)
        ground_truth = np.sum(speed_obs >= 0)[1] > self.K/2 

        space_obs = np.copy(speed_obs)
        for i in range(space_obs):
            space_obs[i] += np.random.normal

        trial = {'ground_truth': ground_truth,
                    'speed' : speed_obs,
                    'place' : space_obs}
        trial.update(kwargs)  # allows wrappers to modify the trial

        # Adding periods sequentially
        self.add_period(['fixation', 'delay', 'decision'])

        # Setting observations, default all 0
        # Setting fixation cue to 1 before decision period
        self.add_ob(space_obs, where='fixation')

        # Setting ground-truth value for supervised learning
        self.set_groundtruth(ground_truth, 'decision')

        self.speed_obs = speed_obs
        self.space_obs = space_obs

        return trial

    def _step(self, action):
        """
        _step receives an action and returns:
            a new observation, obs
            reward associated with the action, reward
            a boolean variable indicating whether the experiment has end, done
            a dictionary with extra information:
                ground truth correct response, info['gt']
                boolean indicating the end of the trial, info['new_trial']
        """
        self.total_steps += 1
        # rewards
        reward = 0
        gt = 0
        done = self.max_step > self.total_steps

        # Get prediction based on action
        pred = 0
        for i in range(len(self.space_obs)):

            gt += (self.speed_obs[i] > 0)
            if abs(action - self.space_obs[i]) < max(self.total_steps, 7):
                pred += (self.speed_obs[i] > 0)

        if pred > 0 == gt > 0:  # if correct
            reward = self.rewards['correct']

        else:  # if incorrect
            reward = self.rewards['fail']
            for i in range(len(self.space_obs)):
                if abs(action - self.space_obs[i]) < max(self.total_steps, 7):
                    self.speed_obs[i] *= -1

        # Update state
        for i in range(len(self.space_obs)):
            self.space_obs[i] += self.speed_obs[i]

        return self.space_obs, reward, done, {'new_trial': new_trial, 'gt': gt}


class DawTwoStep(ngym.TrialEnv):
    """On each trial, an initial choice between two options lead to either of two, second-stage states. 
    In turn, these both demand another two-option choice, each of which is associated with a different chance of receiving reward.

    https://neurogym.github.io/envs/DawTwoStep-v0.html"""

    pass
