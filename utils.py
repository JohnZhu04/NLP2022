import pickle

def save_pkl(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    print('Saved to {}'.format(path))

def load_pkl(path):
    print('Loading from {}'.format(path))
    with open(path, 'rb') as f:
        return pickle.load(f)