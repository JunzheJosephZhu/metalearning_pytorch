import torch
device = 0
class A3C(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.init_h = torch.nn.Parameter(torch.zeros(1, 48)).cuda(device)
        self.init_c = torch.nn.Parameter(torch.zeros(1, 48)).cuda(device)
        self.lstm = torch.nn.LSTMCell(4, 48) # first two are previous action, then reward and cost
        self.action_head = torch.nn.Linear(48, 2)
        self.value_head = torch.nn.Linear(48, 1)
    def forward(self, prev_action, prev_reward, prev_cost, lstm_state):
        input_vec = torch.zeros(4).cuda(device)
        if prev_action!=-1: # if not first trial
            input_vec[prev_action] = 1
        input_vec[2] = prev_reward
        input_vec[3] = prev_cost
        if lstm_state == None:
            h, c = self.lstm(input_vec[None, ...], (self.init_h, self.init_c))
        else:
            h, c = self.lstm(input_vec[None, ...], lstm_state)
        action_probs = self.action_head(h).softmax(1).clamp(min = 1e-8)
        value = self.value_head(h)
        return action_probs, value, (h, c)
