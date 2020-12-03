import numpy as np
def discount(rewards, discount_f):
    cumsum = 0
    value_targets = []
    for i in range(0, len(rewards)):
        index = - i - 1 # starting at -1, ending at 0
        cumsum = cumsum * discount_f + rewards[index]
        value_targets = [cumsum] + value_targets
    return value_targets

def inverse_sigmoid(y):
    x = np.log(y/(1-y))
    return x

if __name__ == "__main__":
    print(inverse_sigmoid(1))