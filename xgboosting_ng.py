import xgboost as xgb
import numpy as np
import tools.eval
import time

t = time.time()

RETRAIN = True
TEST = False
DATA_DIR = 'C:/ml_data'
OUTPUT_DIR = './output/'
MODEL_DIR = './model/xgb'


train_test_margin = 47000
X_train = []
X_test = []
for feature in range(3):
    X_train.append(np.load(f'{DATA_DIR}/important{feature}_alpha.npy' , mmap_mode='r'))
    X_test.append(X_train[feature][train_test_margin:, :])
    X_train[feature] = X_train[feature][0:train_test_margin, :]

y_train = np.load(f'{DATA_DIR}/y_train.npy', mmap_mode='r')
y_test = y_train[train_test_margin:, :]
y_train = y_train[0:train_test_margin, :]

np.save(f'{DATA_DIR}/y_in.npy', y_train)
np.save(f'{DATA_DIR}/y_out.npy', y_test)


experiments = [1]
result = []
for var in experiments:
    var = var
    print("\nExperimenting var =", var)

    submissions_in = []
    submissions_out = []
    submissions = []
    for feature in range(3):
        bst = None
        x_ = X_train[feature]
        y_ = y_train[:,feature]
        param = {'objective': 'reg:tweedie', 'booster': 'dart', 'num_parallel_tree': 5, 
                'lambda' : 1, 'alpha': 0, 'subsample': 1, 'max_depth': 10}

        if RETRAIN:
            print(f'Training feature {feature}')
            size = x_.shape[0]
            batchSize = 10000

            bst = None
            iters = size // batchSize + 1
            for i in range(iters):
                print(f'(feature {feature}) Iteration {i + 1} of {iters}:', end='\r')
                dTrain = xgb.DMatrix(x_[i*batchSize: (i+1)*batchSize], 
                                label = y_[i*batchSize:(i+1)*batchSize])
                #print('Training xgboost')
                bst = xgb.train(param, dTrain, xgb_model=bst, num_boost_round=12)
                dTrain = None

            bst.save_model(f'{MODEL_DIR}/xgboost{feature}.model')

        else:
            bst = xgb.Booster(param)
            bst.load_model(f'{MODEL_DIR}/xgboost{feature}.model')

        print(f"Predicting feature: {feature}")
        if TEST : #to be fixed
            x_ = np.load(f'{DATA_DIR}/important_test{feature}_alpha.npy', mmap_mode='r')
            test = xgb.DMatrix(x_)
            ypred = bst.predict(test)
            submissions.append(ypred)
        
        
        test_in = xgb.DMatrix(X_train[feature])
        ypred_in = bst.predict(test_in)
        submissions_in.append(ypred_in)

        test_out = xgb.DMatrix(X_test[feature])
        ypred_out = bst.predict(test_out)
        submissions_out.append(ypred_out)


    submissions = np.array(submissions)
    submissions = submissions.T
    np.savetxt(f'{OUTPUT_DIR}/submission.csv', submissions, delimiter=',')

    submissions_in = np.array(submissions_in)
    submissions_in = submissions_in.T
    np.savetxt(f'{OUTPUT_DIR}/submission_in.csv', submissions_in, delimiter=',')

    submissions_out = np.array(submissions_out)
    submissions_out = submissions_out.T
    np.savetxt(f'{OUTPUT_DIR}/submission_out.csv', submissions_out, delimiter=',')

    if not TEST:
        s1 = "E_in: \n%s" % tools.eval.CalcError('y_in.npy', 'submission_in.csv')
        s2 = "E_out: \n%s" % tools.eval.CalcError('y_out.npy', 'submission_out.csv')
        result.append("%s\n%s\n" % (s1, s2))

t = time.time() - t
print('total: ', t, 'sec')

for i in range(len(experiments)):
    print(experiments[i])
    print(result[i])
