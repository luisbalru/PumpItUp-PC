import preprocessing
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn import preprocessing

from sklearn.model_selection import train_test_split
from dml.lmnn import KLMNN

import autoencoder

import numpy as np

from ipf import IPF

from sklearn.feature_selection import SelectKBest, f_classif

from imblearn.under_sampling import EditedNearestNeighbours

from ssma import SSMA

from anomaly_cleaning import cleanAnomalies

def Pipeline(X_train, y_train, X_test):
    id_train = np.array(X_train["id"])
    X_train = X_train.drop(columns=["id"])
    id_test = np.array(X_test["id"])
    X_test = X_test.drop(columns=["id"])

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)

    print("Scaling data...")
    X_train = preprocessing.scale(X_train)
    X_test = preprocessing.scale(X_test)

    '''
    print("PCA...")
    X_train = PCA().fit_transform(X_train)
    X_test = PCA(n_components=len(X_train[0])).fit_transform(X_test)
    '''


    print("IPF...")
    X_train, y_train = IPF(X_train, y_train)
    print("Numero de instancias: " + str(len(X_train)))
    print("Instancias por clase:")
    print(np.unique(y_train,return_counts=True))


    '''
    print("Cleaning anomalies...")
    X_train, y_train = cleanAnomalies(X_train, y_train)
    print("Instancias por clase:")
    print(np.unique(y_train,return_counts=True))
    '''

    '''
    print("Feature selection...")
    feature_selector = SelectKBest(f_classif, k="all").fit(X_train, y_train)
    X_train = feature_selector.transform(X_train)
    X_test = feature_selector.transform(X_test)
    print("Numero de features: " + str(len(X_train[0])))
    '''

    '''
    print("SMOTE...")
    X_train,y_train = SMOTE(random_state=123456789, n_jobs=8).fit_resample(X_train,y_train)
    print("Numero de instancias: " + str(len(X_train)))
    print("Instancias por clase:")
    print(np.unique(y_train,return_counts=True))
    '''


    '''
    print("EditedNearestNeighbours...")
    X_train, y_train = EditedNearestNeighbours(n_neighbors=7, n_jobs=8).fit_resample(X_train, y_train)
    print("Numero de instancias: " + str(len(X_train)))
    print("Instancias por clase:")
    print(np.unique(y_train,return_counts=True))
    '''

    '''
    print("Reduccion de dimensionalidad con AutoEncoder...")
    hid = [256,128,64]
    X_train, X_test = autoencoder.fitTransform(X_train, X_test, 250, hid, bsize=32)
    '''

    '''
    print("SSMA...")
    selector = SSMA(n_neighbors=7, alpha=1, max_loop=250, initial_density=0.9).fit(X_train,y_train)
    X_train = selector.X_
    y_train = selector.y_
    print("Numero de instancias: " + str(len(X_train)))
    '''

    '''
    print("Generando la métrica con DML...")
    train_set, _, train_labels, _ = train_test_split(X_train, y_train, train_size=0.25, random_state=123456789)
    print("Tamaño del conjunto original: " + str(len(X_train)) + ", tamaño del train: " + str(len(train_set)))
    dml = KLMNN(k=7).fit(train_set, train_labels)
    X_train = dml.transform(X_train)
    X_test = dml.transform(X_test)
    '''

    return X_train, y_train, id_train, X_test, id_test
