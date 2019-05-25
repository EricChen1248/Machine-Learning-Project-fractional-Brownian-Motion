import numpy as np
import os

FILES = ["./data/X_train.npz", "./data/Y_train.npz", "./data/X_test.npz"]

for FILE in FILES:
    NPY = FILE[:-3] + "npy"
    if os.path.exists(NPY):
        print(f"{NPY} already exists, skipping {FILE}")
        continue
    print(f"Converting {FILE}")
    x = np.load(FILE, mmap_mode='r+')['arr_0']
    np.save(NPY, x)
    print(f"Finished converting {FILE}")