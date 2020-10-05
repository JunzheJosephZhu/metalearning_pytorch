from experiment import experiment1
from A3C import A3C
from run_session import run_session
import torch
import random
from tqdm import tqdm
import os
import argparse
import json5
from pathlib import Path

torch.manual_seed(0)
random.seed(0)
root = "/home/joseph/Desktop/metalearning_pytorch/per_trial_model"

# options
parser = argparse.ArgumentParser(description='resume')
parser.add_argument('-C', '--configuration', type=str)
parser.add_argument('-R', '--resume', action='store_true')
args = parser.parse_args()

with open(args.configuration) as file:
    configuration = json5.load(os.path.join(root, 'configs', args.configuration))
    config_name = args.configuration.split('.')[0]

experiment_dir = Path(root) / "experiment" / config_name



# intialize agent and load weights
agent = A3C().cuda(0)
optimizer = torch.optim.Adam(agent.parameters(), lr=0.0007)

# train agent cont1
print('training with cont1x blocks')
if os.path.exists(experiment_dir / 'models/cont1x.pth')) and args.resume:
    pkg = torch.load(experiment_dir / 'models/cont1x.pth'), map_location=torch.device('cuda:0'))
    agent.load_state_dict(pkg['state_dict'])
    start_session = pkg['train_session']
    print('load model at %d' % (start_session))
else:
    start_session = 0
for i in tqdm(range(start_session, 15000)):
    experiment = experiment1(0)
    blocks = run_session(experiment, agent, optimizer, suppress_idx=[])
    if i % 100 == 99:
        torch.save({'state_dict':agent.state_dict(), 'train_session':i + 1}, experiment_dir / 'models/cont1x.pth'))

# train agent cont2
print('training with stage2 blocks')
if os.path.exists(experiment_dir / 'models/stage2.pth')) and args.resume:
    pkg = torch.load(os.path.join(experiment_dir / 'models/stage2.pth'), map_location=torch.device('cuda:0'))
    agent.load_state_dict(pkg['state_dict'])
    start_session = pkg['train_session']
    print('load model at %d' % (start_session))
else:
    start_session = 0
for i in tqdm(range(start_session, 15000)):
    if i % 2 == 0:
        experiment = experiment1(1)
    else:
        experiment = experiment1(2)
    blocks = run_session(experiment, agent, optimizer, suppress_idx=[])
    if i % 100 == 99:
        torch.save({'state_dict':agent.state_dict(), 'train_session':i + 1}, experiment_dir / 'models/stage2.pth'))


# test agent
experiment = experiment1(3)
blocks = run_session(experiment, agent, optimizer, train=False, suppress_idx=[])
for block in blocks:
    print(block['type'], block['actions'])
    # print(block['rewards'], block['costs'])