import numpy as np
from sklearn.ensemble import RandomForestRegressor
from joblib import dump, load
import matplotlib.pyplot as plt

VERBOSE = True
DATA_DIR = 'C:/ml_data'
MODEL_DIR = './model/RF'



def genImportantData(threshold = 0.0001):

    print("Threshold:", threshold)

    # load alpha
    if VERBOSE:
        print('loading pre-trained result: alpha...', end = '')
    alpha = np.load(f'{DATA_DIR}/train_alpha.npy', mmap_mode='r')
    test_alpha = np.load(f'{DATA_DIR}/test_alpha.npy' , mmap_mode='r')
    if VERBOSE:
        print('Done.')
    
    # load data
    if VERBOSE:
        print('loading origin data...', end = '')
    X_train = np.load(f'{DATA_DIR}/X_train.npy')
    X_test = np.load(f'{DATA_DIR}/X_test.npy')
    if VERBOSE:
        print('Done.')


    for feature in range(3):
        if VERBOSE:
            print(f'generating feature {feature}...', end = '')

        # load importance
        THRESH = threshold
        rfRegressor = load(f'{MODEL_DIR}/randomForest{feature}.joblib')
        importance = rfRegressor.feature_importances_
        importance = np.where(importance > THRESH)

        train = X_train.T[importance]
        test = X_test.T[importance]
        
        train = train.T
        test = test.T
        train = np.c_[train, alpha]
        test = np.c_[test, test_alpha]

        if VERBOSE:
            print('saving...', end = '')
        np.save(f'{DATA_DIR}/important{feature}_alpha.npy', train)
        np.save(f'{DATA_DIR}/important_test{feature}_alpha.npy', test)
        
        if VERBOSE:
            print('Done.')



if __name__ == '__main__':
    threshold = 0.001
    genImportantData(threshold)