import numpy as np


def CalcError():
    ret = ""
    y = np.load('./data/Y_train.npy')
    p = np.loadtxt('./data/submission.csv', delimiter=',')

    # Feature 1
    w = [300, 1, 200]
    err = np.absolute(y - p) * w
    fet = np.sum(err, axis=0) / y.shape[0]
    res = np.sum(err) / y.shape[0]

    ret += f"Track 1: {fet} {res}\n"


    err = np.absolute(y - p) / y
    fet = np.sum(err, axis=0) / y.shape[0]
    res = np.sum(err) / y.shape[0]

    ret += f"Track 2: {fet} {res}\n"

    return ret


if __name__ == "__main__":
    print(CalcError())