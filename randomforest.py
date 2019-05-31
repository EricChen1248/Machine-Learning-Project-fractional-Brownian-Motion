import numpy as np
from sklearn.ensemble import RandomForestRegressor
from joblib import dump, load


RETRAIN = True
TEST = True
LOAD = True
parms = {'bootstrap': True, 'n_jobs': -1, 'n_estimators': 50, 'max_depth': 10, 'verbose' : 3}
N = [350]

results = []
for i in range(len(N)):
    submissions = []
    for feature in range(3):
        rfRegressor = RandomForestRegressor(**parms)

        if LOAD:
            try:
                rfRegressor = load(f'./.cache/randomForest{feature}.joblib')
                rfRegressor.set_params(**parms)
                rfRegressor.set_params(warm_start = True, n_estimators = N[i])
            except FileNotFoundError:
                pass

        x = np.load('./data/train_avg.npy', mmap_mode='r')
        y = np.load('./data/Y_train.npy', mmap_mode='r')
        y = y[:,feature]

        if RETRAIN:
            rfRegressor.fit(x, y)
            dump(rfRegressor, f'./.cache/randomForest{feature}.joblib')
        else:
            rfRegressor = load(f'./.cache/randomForest{feature}.joblib')

        if TEST:
            x = np.load('./data/test_avg.npy', mmap_mode='r')
        submissions.append(rfRegressor.predict(x))

        '''
        import matplotlib.pyplot as plt
        testy = y
        m = testy.max()
        plt.scatter(testy, ypred)
        plt.plot([0,m],[0, m])
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()
        '''

    submissions = np.array(submissions)
    submissions = submissions.T
    np.savetxt('./data/submission.csv', submissions, delimiter=',')

    import tools.eval
    results.append(tools.eval.CalcError())

for r in results:
    print(r)



