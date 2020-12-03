import sys
sys.path.append('per_trial_model')
from load_pkl import load_blocks
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
root = "/home/joseph/Desktop/metalearning_pytorch"


experiments = {"experiment/config5/2020-11-03-00-22-18.pkl": 0.9,
                "experiment/config6/2020-10-22-10-49-48.pkl": 0.1,
                "experiment/config7/2020-10-22-23-01-36.pkl": 0.5,
                "experiment/config8/2020-10-23-04-37-52.pkl": 0.7,
                "experiment/config9/2020-10-26-15-42-06.pkl": 0.6,
                "experiment/config10/2020-10-26-22-35-32.pkl": 0.8,
                "experiment/config11/2020-10-27-03-40-16.pkl": 0.0}


def compute_indiff_sem(filename):
    blocks = load_blocks(os.path.join(root, filename))

    actions = blocks[1]['actions']
    timesteps = np.arange(len(actions))[:, np.newaxis]
    model = LogisticRegression().fit(timesteps, actions)

    grid_x = np.arange(0, len(actions), 0.01)
    grid_y = model.predict_proba(grid_x[:, np.newaxis])[:, 1]
    # plt.plot(grid_x, grid_y)
    # plt.show(block=False)
    # plt.pause(3)
    # plt.close()
    indifference = grid_x[np.searchsorted(grid_y, 0.5)]
    proba_y = model.predict_proba(timesteps)[:, 1]
    sem = np.sqrt(np.sum(np.square(proba_y - actions)) / actions.shape[0] / (actions.shape[0] - 1))
    return indifference, sem

fig = plt.figure(figsize=(12, 5))
fig.suptitle('performance on a cont1x block, with bv=0.0125')
for filename, discount_f in experiments.items():
    indiff, sem = compute_indiff_sem(filename)
    plt.subplot(121)
    plt.plot(discount_f, indiff, 'bx')
    plt.xlabel('discount_f')
    plt.ylabel('indifference point of sigmoidal fit')
    plt.subplot(122)
    plt.plot(discount_f, sem, 'rx')
    plt.xlabel('discount_f')
    plt.ylabel('bootstrapped SEM')
# plt.show(block=False)
# plt.pause(3)
# plt.close()
plt.savefig('plots/reg_lambda_switchtime.png')