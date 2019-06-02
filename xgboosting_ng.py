import xgboost as xgb
import numpy as np
import tools.eval


RETRAIN = True
TEST = False
DATA_DIR = 'C:/ml_data'
OUTPUT_DIR = './output/'
MODEL_DIR = './model/xgb'


x = []
for feature in range(3):
    x.append(np.load(f'{DATA_DIR}/important{feature}_alpha.npy' , mmap_mode='r'))
    print(x[feature].shape)
y = np.load(f'{DATA_DIR}/y_train.npy', mmap_mode='r')

submissions = []
for feature in range(3):
    bst = None
    x_ = x[feature]
    y_ = y[:,feature]
    param = {'booster':'dart', 'lambda' : 1, 
            'alpha': 0, 'subsample': 1, 'max_depth': 20, 'tree_method': 'gpu_hist'}

    if RETRAIN:
        print(f'Training feature {feature}')
        size = x_.shape[0]
        batchSize = 10000

        bst = None
        iters = size // batchSize + 1
        for i in range(iters):
            print(f'(feature {feature}) Iteration {i + 1} of {iters}:')
            dTrain = xgb.DMatrix(x_[i*batchSize: (i+1)*batchSize], 
                            label = y_[i*batchSize:(i+1)*batchSize])
            print('Training xgboost')
            bst = xgb.train(param, dTrain, xgb_model=bst)
            dTrain = None

        bst.save_model(f'{MODEL_DIR}/xgboost{feature}.model')

    else:
        bst = xgb.Booster(param)
        bst.load_model(f'{MODEL_DIR}/xgboost{feature}.model')

    print(f"Predicting feature: {feature}")
    if TEST :
        x_ = np.load(f'{DATA_DIR}/important_test{feature}_alpha.npy' , mmap_mode='r')
    else:
        x_ = np.load(f'{DATA_DIR}/important{feature}_alpha.npy' , mmap_mode='r')
    test = xgb.DMatrix(x_)
    ypred = bst.predict(test)
    test = None

    submissions.append(ypred)


submissions = np.array(submissions)
submissions = submissions.T
np.savetxt(f'{OUTPUT_DIR}submission.csv', submissions, delimiter=',')

if not TEST:
    print(tools.eval.CalcError())

