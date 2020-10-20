# actions = [a0, a1, a2, ...., aT]
# value_preds = [V0, V1, V2, ...., VT]
# rewards = [r0, r1, r2, ..... , rT]
# value_target = []
import torch
import numpy as np
from utils import discount
device = 0

def run_session(experiment, agent, optimizer, lstm_init = None, train = True, bv = 0.05, be = 0.05, discount_f = 0.9, suppress_idx=[]):
    """
        args:
            experiment: pull_lever() takes input and gives output
                        blocks: each a dict of {'st', 'ed', 'type'}
    """
    action = -1
    reward, cost = 1, 1 # this is an arbitrary number for the initial input
    lstm_state = lstm_init
    # all lists are one item per trial
    value_preds = [] 
    rewards = []
    costs = []
    actions = []
    policies = []
    activ_hist = [] # list of hidden states

    # forward pass
    while not experiment.terminated:
        action_probs, value_pred, (hidden_state, _) = agent(action, reward, cost, lstm_state)
        if len(suppress_idx) > 0:
            hidden_state[0][suppress_idx] = hidden_state[0][suppress_idx] - 0.3
        action = torch.multinomial(action_probs.detach().cpu(), 1)[0]
        reward, cost = experiment.pull(action)
        policies.append(action_probs)
        value_preds.append(value_pred)
        actions.append(action)
        costs.append(cost)
        rewards.append(reward)
        activ_hist.append(hidden_state)
    
    # gradient descent
    rewards, costs = np.array(rewards), np.array(costs)
    value_targets = torch.Tensor(discount(rewards / costs, discount_f)).cuda(device)
    value_preds = torch.Tensor(value_preds).cuda(device)
    policies = torch.cat(policies, dim=0)
    actions = torch.cat(actions, dim=0).long()
    advantage = (value_targets - value_preds).detach()
    activ_hist = torch.cat(activ_hist, dim=0)
    if train:
        value_loss = torch.nn.MSELoss()(value_preds, value_targets)
        policy_loss = -torch.mean(torch.log(policies[torch.arange(policies.size(0)), actions]) * advantage)
        entropy_loss = torch.mean(torch.log(policies) * policies)
        loss = policy_loss + value_loss * bv + entropy_loss * be
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    # collect data
    actions = actions.numpy()
    value_preds = value_preds.detach().cpu().numpy()
    activ_hist = activ_hist.detach().cpu().numpy()
    for i, block in enumerate(experiment.blocks):
        st = block['st']
        # ed = block['ed'] + 1
        ed = experiment.blocks[i + 1]['st'] if i + 1 < len(experiment.blocks) else len(actions)
        block['actions'] = actions[st:ed]
        block['rewards'] = rewards[st:ed]
        block['costs'] = costs[st:ed]
        block['value_preds'] = value_preds[st:ed]
        block['activ'] = activ_hist[st:ed]

    return experiment.blocks, actions, activ_hist

if __name__ == "__main__":
    from experiment import experiment1
    from A3C import A3C
    import torch
    import random
    from tqdm import tqdm
    torch.manual_seed(0)
    random.seed(0)
    agent = A3C().cuda(0)
    agent.load_state_dict(torch.load('experiment/config1/models/stage2.pth', map_location=torch.device('cuda: 0'))['state_dict'])
    optimizer = torch.optim.Adam(agent.parameters(), lr=0.0007)
    for i in tqdm(range(1)):
        experiment = experiment1(0)
        blocks, actions, activ = run_session(experiment, agent, optimizer, suppress_idx=[])
    experiment = experiment1(3)
    blocks, actions, activ = run_session(experiment, agent, optimizer, train=False, suppress_idx=[])
    print(len(actions))
    for block in blocks:
        print(block['type'])
        print(block['actions'])
        print(block['costs'])