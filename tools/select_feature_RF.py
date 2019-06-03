import numpy as np
from sklearn.ensemble import RandomForestRegressor
from joblib import dump, load
import matplotlib.pyplot as plt

DATA_DIR = 'C:/ml_data'
MODEL_DIR = './model/RF'


alpha = np.load(f'{DATA_DIR}/train_alpha.npy', mmap_mode='r')
test_alpha = np.load(f'{DATA_DIR}/test_alpha.npy' , mmap_mode='r')

for feature in range(3):
    THRESH = 0.0001
    rfRegressor = load(f'{MODEL_DIR}/randomForest{feature}.joblib')
    importance = rfRegressor.feature_importances_
    importance = np.where(importance > THRESH)
    #print("Threshold",THRESH)
    #importance=np.array(importance)
    #print(importance.size)
    train = np.load(f'{DATA_DIR}/X_train.npy').T[importance]
    test = np.load(f'{DATA_DIR}/X_test.npy').T[importance]
    train = train.T
    test = test.T
    #print(train.shape, alpha.shape)
    #exit()
    train = np.c_[train, alpha]
    test = np.c_[test, test_alpha]
    np.save(f'{DATA_DIR}/important{feature}_alpha.npy', train)
    np.save(f'{DATA_DIR}/important_test{feature}_alpha.npy', test)
