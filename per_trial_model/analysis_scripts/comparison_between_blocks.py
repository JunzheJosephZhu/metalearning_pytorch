import sys
sys.path.append('per_trial_model')
from load_pkl import load_blocks
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
# import seaborn as sns; sns.set_theme(color_codes=True)
import pandas as pd
from utils import inverse_sigmoid

pool = True # pool all blocks of the same type
filename = "/home/joseph/Desktop/metalearning_pytorch/experiment/config8/2020-10-23-04-37-52.pkl"
blocks = load_blocks(filename)

type2idx = {'cont1x':0, 'cont2x':1, 'disc1x':2, 'disc2x':3}
type_plotted = set()
fig = plt.figure(figsize=(20, 12))
outer = gridspec.GridSpec(2, 2, wspace=0.2, hspace=0.4)
inners = [gridspec.GridSpecFromSubplotSpec(2, 1,
                        subplot_spec=outer[i], wspace=0.2, hspace=0.4)
                        for i in range(4)]
inner_subplots = [[plt.Subplot(fig, inner[0]), plt.Subplot(fig, inner[1])]
                     for inner in inners]
for block in blocks:
    i = type2idx[block['type']]
    if block['type'] not in type_plotted or pool:
        type_plotted.add(block['type'])
        inner = inners[i]
        timesteps = np.arange(len(block['actions']))
        ax = inner_subplots[i][0]
        ax.scatter(timesteps, block['costs'], c=block['actions'])

        ax = inner_subplots[i][1]
        ax.scatter(timesteps, block['rewards'], c=block['actions'])

for [ax1, ax2], name in zip(inner_subplots, list(type2idx.keys())):
    fig.add_subplot(ax1)
    plt.ylabel('Presses', fontsize=16)
    plt.title(name, fontsize=16)
    plt.xlim(0, 60)
    plt.ylim(0, 50)
    fig.add_subplot(ax2)
    plt.xlabel('Trial', fontsize=16)
    plt.ylabel('Reward', fontsize=16)
    plt.xlim(0, 60)

plt.savefig('plots/comparison_between_blocks.png')