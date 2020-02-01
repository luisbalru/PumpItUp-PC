import preprocessing
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from sklearn import preprocessing

import autoencoder

import numpy as np

def Pipeline(X_train, y_train, X_test):
    id_train = np.array(X_train["id"])
    X_train = X_train.drop(columns=["id"])
    id_test = np.array(X_test["id"])
    X_test = X_test.drop(columns=["id"])

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)

    X_train = preprocessing.scale(X_train)
    X_test = preprocessing.scale(X_test)

    #hid = [1024,512,256,128]
    #X_train, X_test = autoencoder.fitTransform(X_train, X_test, 10, hid, bsize=32)

    '''
    X = PCA(n_components=700).fit_transform(X)

    if train:
        X,y = SMOTE(random_state=123456789).fit_resample(X,y)
    '''
    return X_train, y_train, id_train, X_test, id_test
