import xgboost as xgb
import numpy as np

# feature = 2
RETRAIN = True
TEST = False

submissions = []
for feature in range(3):
    x = None
    y = None
    bst = None
    x = np.load('./data/X_train.npy', mmap_mode='r')
    y = np.load('./data/Y_train.npy', mmap_mode='r')
    y = y[:,feature]
    param = {'booster':'gblinear', 'lambda' : 1, 'alpha': 0, 'subsample': 1, 'predictor' : 'cpu_predictor', 'max_depth': 20}

    if RETRAIN:
        print(f'Training feature {feature}')
        size = x.shape[0]
        batchSize = 10000

        bst = None
        iters = size // batchSize + 1
        for i in range(iters):
            print(f'Iteration {i + 1} of {iters}:')
            dTrain = xgb.DMatrix(x[i*batchSize: (i+1)*batchSize], label = y[i*batchSize:(i+1)*batchSize])
            print('Training xgboost')
            bst = xgb.train(param, dTrain, xgb_model=bst)
            dTrain = None

        bst.save_model(f'./.cache/xgboost{feature}.model')

    else:
        bst = xgb.Booster(param)
        bst.load_model(f'./.cache/xgboost{feature}.model')

    print(f"Predicting feature: {feature}")
    if TEST :
        x = np.load('./data/X_test.npy', mmap_mode='r')
    else:
        x = np.load('./data/X_train.npy', mmap_mode='r')
    test = xgb.DMatrix(x)
    ypred = bst.predict(test)
    test = None

    '''
    import matplotlib.pyplot as plt
    testy = y[:predictionSize]
    m = testy.max()
    plt.scatter(testy, ypred)
    plt.plot([0,m],[0, m])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
    '''
    submissions.append(ypred)


submissions = np.array(submissions)
submissions = submissions.T
np.savetxt('./data/submission.csv', submissions, delimiter=',')


