import xgboost as xgb
import numpy as np

feature = 2
RETRAIN = False

submissions = []
for feature in range(3):
    x = None
    y = None
    bst = None
    x = np.load('./data/X_train.npy', mmap_mode='r')
    y = np.load('./data/Y_train.npy', mmap_mode='r')
    y = y[:,feature]
    param = {'booster':'gblinear', 'lambda' : 1, 'alpha': 0, 'subsample': 0.5, 'verbosity' : 2, 'predictor' : 'cpu_predictor', 'max_depth': 20}

    if RETRAIN:
        print('Loading Data')
        dTrain = xgb.DMatrix(x, label = y)

        x = None
        y = None
        print('Training xgboost')
        bst = None
        bst = xgb.train(param, dTrain, 5)
        bst.save_model(f'./.cache/train{feature}.model')

        dTrain = None

    else:
        bst = xgb.Booster(param)
        bst.load_model(f'./.cache/train{feature}.model')

    print(f"Predicting feature: {feature}")
    x = np.load('./data/X_train.npy', mmap_mode='r')
    #y = np.load('./data/Y_train.npy', mmap_mode='r')[:,feature]
    test = xgb.DMatrix(x)
    ypred = bst.predict(test)
    #ypred = np.array(list(map(lambda x : x + (x - 0.5) / 4, ypred)))
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


