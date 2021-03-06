import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import sys

VERBOSE = True
DATADIR = './data'
GEN_TRAIN = True


def average(data):
    N, dim = data.shape
    interval = 10
    data = data.T
    
    new = np.zeros((int(dim/interval), N))
    for dim_i in range(int(dim/interval)):
        new[dim_i] =  np.average(data[dim_i * interval: dim_i * interval + interval])
    
    new = new.T
    return new


# load data
if GEN_TRAIN:
    X = np.load(f"{DATADIR}/X_train.npy", mmap_mode='r')
else:
    X = np.load(f"{DATADIR}/X_test.npy", mmap_mode='r')
N, dim = X.shape

# regenerate
X_new = []
for i in range(N):
    if VERBOSE:
        dots = "."*(int(i/100)%3+1) + " "*(3-(int(i/100)%3+1))
        print("processing %i/%i data" % (i+1, N), dots, end="\r")

    t = np.array(range(int(dim/2)))
    x_i = X[i, 0:int(dim/2)]
    t.shape = (t.shape[0], 1)
    x_i.shape = (x_i.shape[0], 1)
    
    model = LinearRegression(fit_intercept=True)
    model.fit(t, x_i)
    
    y_pred = model.predict(t)
    x_i = x_i - y_pred
    # x_i = x_i * x_i
    x_i.shape = (x_i.shape[0], )

    #  X_new.append(np.r_[x_i, X[i, int(dim/2):]])
    X_new.append(x_i)


if VERBOSE:
    print("generating new X...       ")
X_new = np.array(X_new)
X_new_avg = np.c_[average(X_new[:, :5000]), X_new[:, 5000:]]

if VERBOSE:
    print("New shape of X:", X_new.shape)
    print("New shape of X_avg:", X_new_avg.shape)
    print("saving to npy file...")
if GEN_TRAIN:
    np.save(f'{DATADIR}/X_LinRig_SE_train.npy', X_new)
    np.save(f'{DATADIR}/X_LinRig_SE_avg_train.npy', X_new_avg)
else:
    np.save(f'{DATADIR}/X_LinRig_SE_test.npy', X_new)
    np.save(f'{DATADIR}/X_LinRig_SE_avg_test.npy', X_new_avg)

if VERBOSE:
    print("done.")
