import numpy as np
from keras.models import Sequential
from keras.models import Model
from keras.models import load_model
from keras.layers import Input, Dense, concatenate, Dropout
import matplotlib.pyplot as plt
import tools.eval
from argparse import ArgumentParser
import os
from autoencoder import encoder

DATA_DIR = 'C:/ml_data'
OUTPUT_DIR = './output/'
MODEL_DIR = './model/NNet'



def K_fold(X, train_test_margin=int(47500*4/5)):
    X_in = X[:train_test_margin, :]
    X_out = X[train_test_margin:, :]
    return X_in, X_out



def build():
    model = []
    for feature in range(3):
        input_MSD = Input(shape = (1024, ))
        input_VAC = Input(shape = (1024, ))
        input_pred = Input(shape = (3, ))
        input_that = Input(shape = (1, ))

        merge = concatenate([input_MSD, input_VAC])
        out = Dropout(0.3)(merge)
        out = Dense(16, 
                kernel_initializer='random_normal',
                bias_initializer='random_normal', 
                activation='tanh')(out)
        out = Dropout(0.3)(out)
        merge = concatenate([out, input_pred])
        out = Dense(3, 
                kernel_initializer='zeros',
                bias_initializer='zeros', 
                activation='tanh')(merge)
        merge = concatenate([out, input_that])
        out = Dense(1, 
                kernel_initializer='ones',
                bias_initializer='zeros')(merge)

        input_list = [input_MSD, input_VAC, input_pred, input_that]
        model.append(Model(inputs=input_list, outputs=out))
        model[feature].compile(loss='mse', optimizer='adam')
    return model



# (build, ) train model and print E_in
def train(X, y_predicted, y_train, model=None, iter=250, label='in'): 
    np.save(f'{DATA_DIR}/y_{label}.npy', y_train)
    N, dim = X.shape
    
    X_MSD = X[:, 0:1024]
    X_VAC = X[:, 5000:6024]
    '''
    X_MSD = X[:, 0:5000]
    X_VAC = X[:, 5000:10000]
    X_MSD = encoder(X_MSD, layer_list = (2048, 1024), name = 'X_MSD_5000_1024', use_old=True)
    X_VAC = encoder(X_VAC, layer_list = (2048, 1024), name = 'X_VAC_5000_1024', use_old=True)
    '''

    if model == None:
        model = build()
    elif model == 'LOAD':
        model = [ load_model('model/NNet/model_%d.h5' % feature) for feature in range(3) ]
    
    submissions = []
    for feature in range(3):
        y_predicted_that = y_predicted[:, feature]
        X_train = [X_MSD, X_VAC, y_predicted, y_predicted_that]

        print("------Training feature %d------" % feature)
        total_steps = iter
        for step in range(total_steps):
            if step%50 == 0:
                print('(feature %d) step = %d of %d' % (feature, step, total_steps))
                model[feature].save('model/NNet/model_%d.h5' % feature)
            cost = model[feature].train_on_batch(X_train, y_train[:, feature])
            if step % 10 == 0:
                print('train cost: ', cost)
        
        # predict feature i and add to list
        y_pred = model[feature].predict(X_train)
        y_pred.shape = (y_pred.shape[0], )
        submissions.append(y_pred)
        

    submissions = np.array(submissions)
    submissions = submissions.T
    # save
    np.savetxt(f'{OUTPUT_DIR}/submission_in.csv', submissions, delimiter=',')
    # E_in
    print("E_%s: \n%s" % (label, tools.eval.CalcError(y_name=f'y_{label}.npy', p_name=f'submission_{label}.csv')))
    return model



# predict output and print E_out
def predict(X, y_predicted, y_test=None, model='LOAD', label='out'):
    if label == 'test':
        print('Generating TEST submission!')
    np.save(f'{DATA_DIR}/y_{label}.npy', y_test)
    if model == 'LOAD':
        model = [ load_model('model/NNet/model_%d.h5' % feature) for feature in range(3) ]
    
    X_MSD = X[:, 0:1024]
    X_VAC = X[:, 5000:6024]
    '''
    X_MSD = X[:, 0:5000]
    X_VAC = X[:, 5000:10000]
    X_MSD = encoder(X_MSD, layer_list = (2048, 1024), name = 'X_MSD_5000_1024', use_old=True)
    X_VAC = encoder(X_VAC, layer_list = (2048, 1024), name = 'X_VAC_5000_1024', use_old=True)
    '''
    submissions = []
    for feature in range(3):
        y_predicted_that = y_predicted[:, feature]
        X_train = [X_MSD, X_VAC, y_predicted, y_predicted_that]

        y_pred = model[feature].predict(X_train)
        y_pred.shape = (y_pred.shape[0], )
        submissions.append(y_pred)
    
    submissions = np.array(submissions)
    submissions = submissions.T
    np.savetxt(f'{OUTPUT_DIR}/submission_{label}.csv', submissions, delimiter=',')
    if not label == 'test':
        print("E_%s: \n%s" % (label, tools.eval.CalcError(y_name=f'y_{label}.npy', p_name=f'submission_{label}.csv')))
    return submissions



if __name__ == '__main__':

    X_train = np.load(f'{DATA_DIR}/X_train.npy', mmap_mode='r')
    X_in, X_out = K_fold(X_train)
    X_test = np.load(f'{DATA_DIR}/X_test.npy', mmap_mode='r')

    y_predicted = np.load(f'{DATA_DIR}/y_train_predict.npy', mmap_mode='r')
    y_out_predicted = np.load(f'{DATA_DIR}/y_out_predict.npy', mmap_mode='r')
    y_test_predicted = np.load(f'{DATA_DIR}/y_test_predict.npy', mmap_mode='r')

    y_train = np.load(f'{DATA_DIR}/y_train.npy', mmap_mode='r')
    y_in, y_out = K_fold(y_train)

    
    model = train(X_in, y_predicted, y_in, iter=150)
    y_pred = predict(X_out, y_out_predicted, y_out, model)

    '''
    y1 = np.loadtxt(f'{DATA_DIR}/test_submission1.csv', delimiter=',')
    y_pred1 = (y_pred + y_test_predicted + y1)/3
    np.savetxt(f'{OUTPUT_DIR}/submission_one.csv', y_pred1, delimiter=',')
    #print("E_out1: \n%s" % (tools.eval.CalcError(y_name=f'y_out.npy', p_name=f'submission_one.csv')))
    '''
    exit()