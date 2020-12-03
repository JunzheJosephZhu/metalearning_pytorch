import sys
sys.path.append('per_trial_model')
from load_pkl import load_blocks
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt

filename = "/home/joseph/Desktop/metalearning_pytorch/experiment/config8/2020-10-23-04-37-52.pkl"
blocks = load_blocks(filename)

# First experiment: PR reward / PR cost, FR reward / FR cost -> action
recep_field = 5
X, Y = [], []
for block in blocks:
    for i in range(recep_field, len(block['actions'])):
        PR_mask = block['actions'][i - recep_field: i]
        X.append(np.concatenate([(block['rewards'] / block['costs'])[i - recep_field: i] * PR_mask,
                                (block['rewards'] / block['costs'])[i - recep_field: i] * (1 - PR_mask)
                                ]))
        Y.append(block['actions'][i])
model = LogisticRegression().fit(X, Y)
plt.figure(figsize=(40, 12))

plt.subplot(121)
plt_len = len(blocks[0]['actions']) + len(blocks[1]['actions']) - recep_field * 2 
plt.scatter(np.arange(plt_len), Y[:plt_len], s=360, marker='o')
plt.scatter(np.arange(plt_len), model.predict(X[:plt_len]), s=600, marker='x', linewidths=5)
plt.legend(['truth', 'predictions'], fontsize=30)
plt.ylabel('actions', fontsize=30)
plt.xlabel(f"regression with past {recep_field} PR value=(reward/cost) & FR value=(reward/cost)\n over whole type 3 session only first 2 blocks shown", fontsize=30)

plt.subplot(122)
x_base = np.arange(recep_field) * 2
weights = model.coef_[0]
plt.bar(x_base, height=weights[0: recep_field][::-1], width=0.5)
plt.bar(x_base + 0.5, height=weights[recep_field: recep_field * 2][::-1], width=0.5)
plt.legend(['PR reward / cost', 'FR reward / cost'], fontsize=30, loc='center')
plt.xlabel('# trials from current predicted trial, left is more recent\nFR is 1, PR is 0', fontsize=30)
plt.ylabel('weights', fontsize=30)
plt.title("Note, FR reward/cost is a constant, so you can think of the weights as simply a bias", fontsize=30)

plt.savefig('plots/reg1_2.png')