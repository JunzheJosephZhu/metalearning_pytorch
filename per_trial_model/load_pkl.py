import pickle
filename = "/home/joseph/Desktop/metalearning_pytorch/experiment/debug/2020-10-14-01-56-28.pkl"
def load_blocks(filename):
    with open(filename, 'rb') as file:
        blocks = pickle.load(file)
    return blocks

if __name__ == "__main__":
    blocks = load_blocks(filename)
    for block in blocks:
        print(f"start trial {block['st']} | end trial {block['ed']} | block type {block['type']}")
        print(f"actions {block['actions']}")
        print(f"costs {block['costs']}")
        print(f"reward {block['rewards']}")
        activations = block["activ"]
        print(activations.shape)
        print("\n")