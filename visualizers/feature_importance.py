import numpy as np
from sklearn.ensemble import RandomForestRegressor
from joblib import dump, load
import matplotlib.pyplot as plt

for feature in range(3):
    rfRegressor = load(f'./.cache/randomForest{feature}.joblib')
    plt.plot(rfRegressor.feature_importances_)
    plt.show()