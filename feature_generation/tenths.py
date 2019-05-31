import numpy as np


def average(data):
    new = np.zeros((1000, data.shape[1]))
    for i in range(1_000):
        new[i] =  np.average(data[i * 10: i * 10 + 10])
    new = new.T
    return new

def skipped(data):
    new = np.zeros((1000, data.shape[1]))
    for i in range(1_000):
        new[i] = data[i * 10]

    new = new.T
    return new

train = np.load('./data/X_train.npy').T
test = np.load('./data/X_test.npy').T


np.save('./data/train_avg.npy', average(train))
np.save('./data/test_avg.npy', average(test))

np.save('./data/train_skip.npy', skipped(train))
np.save('./data/test_skip.npy', skipped(test))