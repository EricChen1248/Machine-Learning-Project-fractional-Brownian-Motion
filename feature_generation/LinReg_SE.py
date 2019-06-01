import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import sys
#import tenths
VERBOSE = True
DATADIR = '../../data'


def average(data):
    data = data.T
    new = np.zeros((1000, data.shape[1]))
    for i in range(1_000):
        new[i] =  np.average(data[i * 10: i * 10 + 10])
    new = new.T
    return new


# load data
X = np.load(f"{DATADIR}/X_train.npy", mmap_mode='r')
N, dim = X.shape

# regenerate
X_new = []
for i in range(N):
    if VERBOSE:
        dots = "."*(int(i/100)%3+1) + " "*(3-(int(i/100)%3+1))
        print("processing %i/%i data" % (i, N), dots, end="\r")

    t = np.array(range(int(dim/2)))
    x_i = X[i, 0:int(dim/2)]
    t.shape = (t.shape[0], 1)
    x_i.shape = (x_i.shape[0], 1)
    
    model = LinearRegression(fit_intercept=True)
    model.fit(t, x_i)
    
    y_pred = model.predict(t)
    x_i = x_i - y_pred
    x_i = x_i * x_i
    x_i.shape = (x_i.shape[0], )

    X_new.append(np.r_[x_i, X[i, int(dim/2):]])


X_new = np.array(X_new)
X_new_avg = np.c_[average(X_new), X_new[:, 5000:]]
if VERBOSE:
    print("New shape of X:", X_new.shape)
    print("New shape of X_avg:", X_new_avg.shape)
    print("saving to npy file...")

np.save(f'{DATADIR}/X_LinRig_SE.npy', X_new)
np.save(f'{DATADIR}/X_LinRig_SE_avg.npy', X_new_avg)

if VERBOSE:
    print("done.")