import numpy as np
import torch
import torch.nn as nn

import neurogym as ngym

# Environment
task = 'DawTwoStep-v0'
kwargs = {'dt': 100}
seq_len = 100

# Make supervised dataset
dataset = ngym.Dataset(task, env_kwargs=kwargs, batch_size=16,
                       seq_len=seq_len)
env = dataset.env
ob_size = env.observation_space.shape[0]
act_size = env.action_space.n

########################
#   BUILDING A MODEL   #
########################

class Net(nn.Module):
    def __init__(self, num_h):
        super(Net, self).__init__()
        self.lstm = nn.RNN(ob_size, num_h)
        self.linear = nn.Linear(num_h, act_size)

    def forward(self, x):
        out, hidden = self.lstm(x)
        x = self.linear(out)
        return x

device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = Net(num_h=64).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)

running_loss = 0.0
for i in range(2000):
    inputs, labels = dataset()
    inputs = torch.from_numpy(inputs).type(torch.float).to(device)
    labels = torch.from_numpy(labels.flatten()).type(torch.long).to(device)

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = net(inputs)

    loss = criterion(outputs.view(-1, act_size), labels)
    loss.backward()
    optimizer.step()

    # print statistics
    running_loss += loss.item()
    if i % 200 == 199:
        print('{:d} loss: {:0.5f}'.format(i + 1, running_loss / 200))
        running_loss = 0.0

print('Finished Training')

########################
#      EVALUATION      #
########################

def eval(env, net, num_trial):

    perf = 0

    for i in range(num_trial):
        env.new_trial()
        ob, gt = env.ob, env.gt
        ob = ob[:, np.newaxis, :]  # Add batch axis
        inputs = torch.from_numpy(ob).type(torch.float).to(device)

        action_pred = net(inputs)
        action_pred = action_pred.detach().numpy()
        action_pred = np.argmax(action_pred, axis=-1)
        perf += gt[-1] == action_pred[-1, 0]

    perf /= num_trial
    return perf

num_trial = 200
print('Average performance in {:d} trials'.format(num_trial))
print(eval(env, net, num_trial))
