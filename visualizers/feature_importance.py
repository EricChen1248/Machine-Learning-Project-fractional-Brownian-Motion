import numpy as np
from sklearn.ensemble import RandomForestRegressor
from joblib import dump, load
import matplotlib.pyplot as plt

for feature in range(3):
    THRESH = 0.001
    rfRegressor = load(f'./.cache/randomForest{feature}.joblib')
    importance = rfRegressor.feature_importances_
    importance = np.where(importance > THRESH)
    
    train = np.load('./data/X_train.npy').T[importance]
    np.save('./data/important.npy', train.T)

