import numpy as np
import torch
import torch.nn as nn

import neurogym as ngym

from tasks import AverageDirectionTest

# Environment
env = AverageDirectionTest()
_ = env.reset()
ob_size = env.observation_space.shape[0]
act_size = env.action_space.n

# CHANGE BELOW TO FIT HOWEVER YOU'VE WRITTEN THE TRAINING METHOD
model = A2C(LstmPolicy, env, verbose=1, policy_kwargs={'feature_extraction':"mlp"})
model.learn(total_timesteps=100000, log_interval=1000)
env.close()
