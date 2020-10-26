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

with open('plots/mean_value_by_block.txt', 'w+') as handle:
    for block in blocks:
        handle.write(block['type'] + ' ')
        handle.write(str(np.mean(block['rewards'] / block['costs'])))
        handle.write('\n')
