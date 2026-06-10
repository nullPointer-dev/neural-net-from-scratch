import numpy as np

def flatten(images):
    return images.reshape(images.shape[0], -1)

def normalize(images):
    return images / 255.0

def one_hot_encode(labels):
    encoded = np.zeros((labels.size, 10))
    for i in range(labels.size):
        encoded[i, labels[i]] = 1
    return encoded