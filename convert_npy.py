import numpy as np

FILES = ["./data/X_train.npz", "./data/Y_train.npz", "./data/X_test.npz"]

for FILE in FILES:
    print(f"Converting {FILE}")
    x = np.load(FILE, mmap_mode='r+')['arr_0']
    np.save(FILE[:-3] + ".npy", x)
    print(f"Finished converting {FILE}")