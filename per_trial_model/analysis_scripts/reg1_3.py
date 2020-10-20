import sys
sys.path.append('per_trial_model')
from load_pkl import load_blocks
from sklearn.linear_model import LogisticRegression, LinearRegression
import numpy as np
import matplotlib.pyplot as plt

filename = "/home/joseph/Desktop/metalearning_pytorch/experiment/debug/2020-10-14-01-56-28.pkl"
blocks = load_blocks(filename)

# First experiment: Neural activation -> action, reward, cost, reward / cost
X, Y1, Y2, Y3, Y4 = [], [], [], [], []
for block in blocks:
    for i in range(len(block['actions'])):
        X.append(block['activ'][i])
        Y1.append(block['actions'][i])
        Y2.append(block['rewards'][i])
        Y3.append(block['costs'][i])
        Y4.append(block['rewards'][i] / block['costs'][i])
model1 = LogisticRegression().fit(X, Y1)
model2 = LinearRegression().fit(X, Y2)
model3 = LinearRegression().fit(X, Y3)
model4 = LinearRegression().fit(X, Y4)
plt_len = len(blocks[0]['actions']) + len(blocks[1]['actions'])

plt.figure(figsize=(40, 10))
plt.subplot(241)
plt.scatter(np.arange(plt_len), Y1[:plt_len], s=90, marker='o')
plt.scatter(np.arange(plt_len), model1.predict(X[:plt_len]), s=150, marker='x', linewidths=2)
plt.legend(['truth', 'predictions'], fontsize=15)
plt.ylabel('actions', fontsize=30)
plt.xlabel(f"regression from neuron", fontsize=30)

plt.subplot(242)
plt.scatter(np.arange(plt_len), Y2[:plt_len], s=90, marker='o')
plt.scatter(np.arange(plt_len), model2.predict(X[:plt_len]), s=150, marker='x', linewidths=2)
plt.legend(['truth', 'predictions'], fontsize=15)
plt.ylabel('rewards', fontsize=30)

plt.subplot(243)
plt.scatter(np.arange(plt_len), Y3[:plt_len], s=90, marker='o')
plt.scatter(np.arange(plt_len), model3.predict(X[:plt_len]), s=150, marker='x', linewidths=2)
plt.legend(['truth', 'predictions'], fontsize=15)
plt.ylabel('costss', fontsize=30)

plt.subplot(244)
plt.scatter(np.arange(plt_len), Y4[:plt_len], s=90, marker='o')
plt.scatter(np.arange(plt_len), model4.predict(X[:plt_len]), s=150, marker='x', linewidths=2)
plt.legend(['truth', 'predictions'], fontsize=15)
plt.ylabel('rewards / costs', fontsize=30)

plt.subplot(245)
weights = model1.coef_[0]
order = np.argsort(weights)
plt.scatter(np.arange(len(weights)), weights[order])
plt.xticks(ticks=np.arange(len(weights)), labels=order, fontsize=6)
plt.xlabel("neuron idx")
plt.ylabel("coefficient for above regression")

plt.subplot(246)
weights = model2.coef_
order = np.argsort(weights)
plt.scatter(np.arange(len(weights)), weights[order])
plt.xticks(ticks=np.arange(len(weights)), labels=order, fontsize=6)
plt.xlabel("neuron idx")
plt.ylabel("coefficient for above regression")

plt.subplot(247)
weights = model3.coef_
order = np.argsort(weights)
plt.scatter(np.arange(len(weights)), weights[order])
plt.xticks(ticks=np.arange(len(weights)), labels=order, fontsize=6)
plt.xlabel("neuron idx")
plt.ylabel("coefficient for above regression")

plt.subplot(248)
weights = model4.coef_
order = np.argsort(weights)
plt.scatter(np.arange(len(weights)), weights[order])
plt.xticks(ticks=np.arange(len(weights)), labels=order, fontsize=6)
plt.xlabel("neuron idx")
plt.ylabel("coefficient for above regression")

plt.title('regression from neurons')
plt.savefig('plots/reg1_3.png')

plt.figure(figsize=(20, 10))
x_base = np.arange(len(model1.coef_[0])) * 2
def minmax(vector):
    return vector / (vector.max() - vector.min())
plt.bar(x=x_base, height=minmax(model1.coef_[0]), width=0.25)
plt.bar(x=x_base + 0.25, height=minmax(model2.coef_), width=0.25)
plt.bar(x=x_base + 0.5, height=minmax(model3.coef_), width=0.25)
plt.bar(x=x_base + 0.75, height=minmax(model4.coef_), width=0.25)
plt.xticks(ticks=x_base, labels=x_base // 2)
plt.title('normalized regression weights by neuron')
plt.legend(["action", "reward", "cost", "reward/cost"])
plt.savefig('plots/reg1_3_extra.png')