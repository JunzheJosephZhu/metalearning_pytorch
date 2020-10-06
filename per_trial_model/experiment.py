import random
import numpy as np
class experiment1():
    def __init__(self, session_type):
        '''
            session_type:
                0: ['cont1x']
                1: ['cont1x', 'cont2x']
                2: ['cont1x', 'disc1x']
                3: ['cont1x', 'cont2x', 'disc1x', 'disc2x']
        '''
        super().__init__()
        # session parameters
        self.session_type = session_type
        self.terminated = False
        self.current_trial = 0 # current trial regardless of block
        self.total_reward = 0

        # current block parameters
        self.consecutive_PR = 0
        self.consecutive_FR = 0
        self.PR_tally = 0
        self.blocks = [{'type':'cont1x'}]

        # experiment parameters
        self.PR_reward = 3
        self.FR_reward = 1
        self.FR_cost = 6
        self.max_reward = 1350 # maximum reward before stopping
        self.max_consecutive_FR = 6 # maximum consecutive FR before switching

    def pull(self, action):
        '''
            action: 0->PR, 1->FR
        '''
        # define PR slope
        current_block = self.blocks[-1]
        block_type = current_block['type']
        block_start = current_block['st'] if 'st' in current_block else self.current_trial
        if block_type == 'cont1x':
            PR_cost_schedule = np.arange(1, 101)
        elif block_type == 'disc1x':
            PR_cost_schedule = np.arange(1, 101)[::2].repeat(2, axis=0)
        elif block_type == 'cont2x':
            PR_cost_schedule = np.arange(1, 200)[::2]
        elif block_type == 'disc2x':
            PR_cost_schedule = np.arange(1, 200)[::4].repeat(2, axis=0)
        else:
            print(f"{block_type} doesn't exist!")
            raise ValueError
        if action == 0:
            reward = self.PR_reward
            cost = int(PR_cost_schedule[self.PR_tally])
            self.PR_tally += 1
            self.consecutive_FR = 0
            self.consecutive_PR += 1
        elif action == 1:
            reward = self.FR_reward
            cost = self.FR_cost
            self.consecutive_PR = 0
            self.consecutive_FR += 1
        else:
            print(f"{action} doesn't exist!")
            raise ValueError
        if 'st' not in current_block:
            if self.consecutive_PR == 4:
                current_block['st'] = self.current_trial - 3
            else:
                self.PR_tally = 0
        if ('st' in current_block and self.consecutive_FR >= self.max_consecutive_FR) or self.current_trial - block_start >= 99:
            self.reset()
        self.current_trial += 1

        self.total_reward += reward
        return reward, cost
    
    def reset(self):
        # 3 is test type, rotated between block types
        self.blocks[-1]['ed'] = self.current_trial
        if self.total_reward < self.max_reward:
            session_type_map = {0: ['cont1x'],
                    1: ['cont1x', 'cont2x'],
                    2: ['cont1x', 'disc1x'],
                    3: ['cont1x', 'disc1x', 'cont1x', 'cont2x', 'cont1x', 'disc2x']}
            block_type_choices = session_type_map[self.session_type]
            if self.session_type != 3:
                self.blocks.append({'type': random.choice(block_type_choices)})
            else:
                current_index = len(self.blocks) % len(block_type_choices)
                self.blocks.append({'type': block_type_choices[current_index]})
            self.PR_tally = 0
            self.consecutive_PR = 0
            self.consecutive_FR = 0
        else:
            self.terminated = True

if __name__ == "__main__":
    experiment = experiment1(0)
    count = 0
    while not experiment.terminated:
        inp = int(input())
        if inp == 2:
            print(experiment.blocks)
        else:
            reward, cost = experiment.pull(inp)
            print(reward, cost)
        count += 1
    print(count)