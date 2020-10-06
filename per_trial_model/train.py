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
import time

torch.manual_seed(0)
random.seed(0)
root = Path("/home/joseph/Desktop/metalearning_pytorch")

# options
parser = argparse.ArgumentParser(description='resume')
parser.add_argument('-C', '--configuration', type=str)
parser.add_argument('-R', '--resume', action='store_true')
args = parser.parse_args()

with open(root / 'configs' / args.configuration) as file:
    configuration = json5.load(file)
    config_name = args.configuration.split('.')[0]
    print(f"using {config_name}")

experiment_dir = root / "experiment" / config_name
model_dir = experiment_dir / "models"
if args.resume:
    assert os.path.exists(model_dir), f"{experiment_dir} doesn't exist"
else:
    os.makedirs(model_dir, exist_ok=True)

date = f"{time.strftime('%Y-%m-%d-%H-%M-%S')}"
with open((experiment_dir / f"{date}.json").as_posix(), "w") as handle:
    json5.dump(configuration, handle, indent=2, sort_keys=False)

# intialize agent and load weights
agent = A3C().cuda(0)
optimizer = torch.optim.Adam(agent.parameters(), lr=0.0007)

# train agent cont1
print('training with cont1x blocks')
if os.path.exists(model_dir / 'cont1x.pth') and args.resume:
    pkg = torch.load(model_dir / 'cont1x.pth', map_location=torch.device('cuda:0'))
    agent.load_state_dict(pkg['state_dict'])
    start_session = pkg['train_session']
    print(f'load model at {model_dir}/cont1x.pth {start_session}')
else:
    start_session = 0
for i in tqdm(range(start_session, 15000)):
    experiment = experiment1(0)
    blocks = run_session(experiment, agent, optimizer, suppress_idx=[], **configuration)
    if i % 100 == 99:
        torch.save({'state_dict':agent.state_dict(), 'train_session':i + 1}, model_dir / 'cont1x.pth')

# train agent cont2
print('training with stage2 blocks')
if os.path.exists(model_dir / 'stage2.pth') and args.resume:
    pkg = torch.load(model_dir / 'stage2.pth', map_location=torch.device('cuda:0'))
    agent.load_state_dict(pkg['state_dict'])
    start_session = pkg['train_session']
    print(f'load model at {model_dir}/stage2.pth {start_session}')
else:
    start_session = 0
for i in tqdm(range(start_session, 15000)):
    if i % 2 == 0:
        experiment = experiment1(1)
    else:
        experiment = experiment1(2)
    blocks = run_session(experiment, agent, optimizer, suppress_idx=[], **configuration)
    if i % 100 == 99:
        torch.save({'state_dict':agent.state_dict(), 'train_session':i + 1}, model_dir / 'stage2.pth')


# test agent
experiment = experiment1(3)
blocks = run_session(experiment, agent, optimizer, train=False, suppress_idx=[], **configuration)
with open(experiment_dir / f"{date}.txt", "w+") as handle:
    for block in blocks:
        print(block['type'], block['actions'])
        handle.write(block['type'] + str(block['actions']) + '\n')
    # print(block['rewards'], block['costs'])

print(list(agent.parameters())[0])