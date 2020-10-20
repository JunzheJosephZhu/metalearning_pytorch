import sys
sys.path.append('per_trial_model')
from load_pkl import load_blocks
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme(color_codes=True)
import pandas as pd

filename = "/home/joseph/Desktop/metalearning_pytorch/experiment/debug/2020-10-14-01-56-28.pkl"
blocks = load_blocks(filename)

# First experiment: PR cost, PR reward, FR cost, FR reward -> action
recep_field = 5
actions = blocks[1]['actions']
data = pd.DataFrame()
data['trial_number'] = np.arange(len(actions))
data['actions'] = actions * 0.99 + 5e-3

sns_plot = sns.lmplot(x="trial_number", y="actions", data=data, logistic=True)
sns_plot.savefig("plots/visualize_actions.png")

print(data)