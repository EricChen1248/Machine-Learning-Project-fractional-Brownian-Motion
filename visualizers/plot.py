import matplotlib.pyplot as plt
import numpy as np

x = np.load("./data/X_train.npy")[0:50, 0:5000]
y = np.load('./data/Y_train.npy')[0:50, 1]

for xs,ys in zip(x, y):
    plt.ylim((0, 0.1))
    plt.plot(xs)
    print(ys)
    plt.show()
