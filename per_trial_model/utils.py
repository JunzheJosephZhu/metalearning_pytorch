def discount(rewards, discount_f):
    cumsum = 0
    value_targets = []
    for i in range(0, len(rewards)):
        index = - i - 1 # starting at -1, ending at 0
        cumsum = cumsum * discount_f + rewards[index]
        value_targets = [cumsum] + value_targets
    return value_targets