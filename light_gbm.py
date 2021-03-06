import lightgbm as lgb
import numpy as np
from joblib import dump, load

SKIP = [False, False, False]
TRAIN = True
OBJECTIVE = 'mae'
ADD = True

x = np.load('./data/X_train.npy')
alpha = np.load('./data/train_alpha.npy', mmap_mode='r+')
x = np.c_[x, alpha]
test = np.load('./data/X_test.npy')
alpha = np.load('./data/test_alpha.npy', mmap_mode='r+')
test = np.c_[test, alpha]
if ADD:
    feature_SE = np.load('./data/X_LinRig_SE_train.npy',mmap_mode='r+')
    x = np.c_[x,feature_SE]
    feature_SE = np.load('./data/X_LinRig_SE_test.npy',mmap_mode='r+')
    test = np.c_[test,feature_SE]
submissions = []
tsubmissions = []
for feature in range(3):
    y = np.load('./data/Y_train.npy', mmap_mode='r+')[:, feature]
    d_train = None
    d_train = lgb.Dataset(x, label=y)

    params = {}
    params['learning_rate'] = 0.01
    params['boosting_type'] = 'gbrt'
    params['objective'] = OBJECTIVE
    params['num_leaves'] = 120
    params['min_data'] = 50
    params['max_depth'] = 20
    params['num_threads'] = 32
    params['verbosity'] = 2
    params['two_round'] = True
    '''
    params['device'] = 'gpu'
    params['gpu_platform_id'] = 0
    params['gpu_device_id'] = 0
    '''
    # params['max_bin'] = 


    print(f'Training feature {feature}')
    if TRAIN:
        clf = None
    else:
        try:
            clf = load(f'./.cache/light{feature}{OBJECTIVE}.joblib')
        except:
            clf = None

    if not SKIP[feature]:
        clf = lgb.train(params, d_train, 1000)

    ypred = clf.predict(x)
    submissions.append(ypred)
    
    dump(clf, f'./.cache/light{feature}{OBJECTIVE}.joblib')

    tpred = clf.predict(test)
    tsubmissions.append(tpred)


submissions = np.array(submissions)
submissions = submissions.T
np.savetxt('./data/submission.csv', submissions, delimiter=',')

tsubmissions = np.array(tsubmissions)
tsubmissions = tsubmissions.T
np.savetxt('./data/test_submission.csv', tsubmissions, delimiter=',')

