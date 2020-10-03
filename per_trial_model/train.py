from experiment import experiment1
from A3C import A3C
from run_session import run_session
import torch
import random
from tqdm import tqdm
import os
import argparse

torch.manual_seed(0)
random.seed(0)
root = "/home/joseph/Desktop/metalearning_pytorch/per_trial_model"

# options
parser = argparse.ArgumentParser(description='resume')
parser.add_argument('-r', '--resume', default=True, help='an integer for the accumulator')
args = parser.parse_args()

# intialize agent and load weights
agent = A3C().cuda(0)
optimizer = torch.optim.Adam(agent.parameters(), lr=0.0007)

# train agent cont1
print('training with cont1x blocks')
if os.path.exists(os.path.join(root, 'models/cont1x.pth')) and args.resume:
    pkg = torch.load(os.path.join(root, 'models/cont1x.pth'), map_location=torch.device('cuda:0'))
    agent.load_state_dict(pkg['state_dict'])
    start_session = pkg['train_session']
    print('load model at %d' % (start_session))
else:
    start_session = 0
for i in tqdm(range(start_session, 15000)):
    experiment = experiment1(0)
    blocks = run_session(experiment, agent, optimizer, suppress_idx=[])
    if i % 100 == 99:
        torch.save({'state_dict':agent.state_dict(), 'train_session':i + 1}, os.path.join(root, 'models/cont1x.pth'))

# train agent cont2
print('training with cont1xcont2x blocks')
if os.path.exists(os.path.join(root, 'models/cont1xcont2x.pth')) and args.resume:
    pkg = torch.load(os.path.join(root, 'models/cont1xcont2x.pth'), map_location=torch.device('cuda:0'))
    agent.load_state_dict(pkg['state_dict'])
    start_session = pkg['train_session']
    print('load model at %d' % (start_session))
else:
    start_session = 0
for i in tqdm(range(start_session, 15000)):
    experiment = experiment1(1)
    blocks = run_session(experiment, agent, optimizer, suppress_idx=[])
    if i % 100 == 99:
        torch.save({'state_dict':agent.state_dict(), 'train_session':i + 1}, os.path.join(root, 'models/cont1xcont2x.pth'))


# test agent
experiment = experiment1(3)
blocks = run_session(experiment, agent, optimizer, train=False, suppress_idx=[])
for block in blocks:
    print(block['actions'], block['rewards'], block['costs'])