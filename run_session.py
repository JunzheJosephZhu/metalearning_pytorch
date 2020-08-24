# actions = [a0, a1, a2, ...., aT]
# value_preds = [V0, V1, V2, ...., VT]
# rewards = [r0, r1, r2, ..... , rT]
# value_target = []
import torch
import numpy as np
from utils import discount
device = 0

def run_session(experiment, agent, optimizer, session_num, lstm_state = None, train = True, bv = 0.05, be = 0.05, discount_f = 0.75):
    be = max(0.05*(20-session_num/1000), be) # input argument be is lower threshold
    action = -1
    reward = 0.25 # this is an arbitrary number for the initial input
    value_preds = []
    rewards = []
    actions = []
    policies = []
    agent.activ_hist = []
    for press_num in range(experiment.num_presses):
        action_probs, value_pred, lstm_state = agent(action, reward, lstm_state)
        agent.activ_hist.append(lstm_state[1][0].detach().cpu().numpy())
        action = np.random.choice([0, 1], p = action_probs.detach().cpu().numpy().flatten())
        reward = experiment.pull(action)
        policies.append(action_probs)
        value_preds.append(value_pred)
        actions.append(action)
        rewards.append(reward)
    value_targets = torch.Tensor(discount(rewards, discount_f)).cuda(device)
    value_preds = torch.Tensor(value_preds).cuda(device)
    policies = torch.cat(policies, dim = 0)
    actions = torch.Tensor(actions).long()
    advantage = (value_targets - value_preds).detach()
    if train:
        value_loss = torch.nn.MSELoss()(value_preds, value_targets)
        policy_loss = -torch.mean(torch.log(policies[torch.arange(policies.shape[0]), actions])*advantage)
        entropy_loss = torch.mean(torch.log(policies)*policies)
        loss = policy_loss + value_loss*bv + entropy_loss*be
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    # collect data
    actions = actions.numpy()
    rewards = np.array(rewards)
    print('session:', session_num, 'reward:', sum(rewards)/len(rewards))
    return actions, rewards, value_preds.cpu().detach().numpy()