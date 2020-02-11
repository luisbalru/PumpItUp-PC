import preprocessing
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import AllKNN
from sklearn import preprocessing
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import plotly.express as px
import plotly

from sklearn.model_selection import train_test_split
from dml.lmnn import KLMNN
from dml.nca import NCA

import autoencoder

import numpy as np
import pandas as pd

from ipf import IPF

from sklearn.feature_selection import SelectKBest, f_classif

from imblearn.under_sampling import EditedNearestNeighbours

from ssma import SSMA

from anomaly_cleaning import cleanAnomalies

colors = ["red", "blue", "green"]

def Pipeline(X_train, y_train, X_test, n_dims=44):
    id_train = np.array(X_train["id"])
    X_train = X_train.drop(columns=["id"])
    id_test = np.array(X_test["id"])
    X_test = X_test.drop(columns=["id"])

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)

    ind_numeric = []
    for i in range(len(X_train[0])):
        if len(np.unique(X_train[:,i]))>2:
            ind_numeric.append(i)

    print("Hay " + str(len(ind_numeric)) + " variables numericas")

    '''
    ind_delete = np.where(y_train=="functional needs repair")[0]
    y_train = np.delete(y_train, ind_delete, axis=0)
    X_train = np.delete(X_train, ind_delete, axis=0)
    '''

    print("Scaling data...")
    X_train = preprocessing.scale(X_train)
    X_test = preprocessing.scale(X_test)


    print("PCA con " + str(n_dims) + " componentes...")
    X_train_binary = np.delete(X_train, ind_numeric, axis=1)
    X_test_binary = np.delete(X_test, ind_numeric, axis=1)
    X_train_numeric = X_train[:,ind_numeric]
    X_test_numeric = X_test[:,ind_numeric]
    pca = PCA(n_components=n_dims)
    X1 = pca.fit_transform(X_train_binary)
    X2 = pca.transform(X_test_binary)
    X_train = np.hstack((X_train_numeric, X1))
    X_test = np.hstack((X_test_numeric, X2))
    print("Numero de features: " + str(len(X_train[0])))


    '''
    print("Reduccion de dimensionalidad con AutoEncoder...")
    hid = [250,200,150,100,50]
    X_train_binary = np.delete(X_train, ind_numeric, axis=1)
    X_test_binary = np.delete(X_test, ind_numeric, axis=1)
    X_train_numeric = X_train[:,ind_numeric]
    X_test_numeric = X_test[:,ind_numeric]
    X1, X2 = autoencoder.fitTransform(X_train_binary, X_test_binary, 30, hid, bsize=32)
    X_train = np.hstack((X_train_numeric, X1))
    X_test = np.hstack((X_test_numeric, X2))
    print("Numero de features: " + str(len(X_train[0])))
    '''

    print("IPF...")
    X_train, y_train = IPF(X_train, y_train)
    print("Numero de instancias: " + str(len(X_train)))
    print("Instancias por clase:")
    print(np.unique(y_train,return_counts=True))


    '''
    print("AllKNN...")
    X_train, y_train = AllKNN(n_neighbors=7, n_jobs=8).fit_resample(X_train, y_train)
    print("Numero de instancias: " + str(len(X_train)))
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


    print("SMOTE...")
    X_train,y_train = SMOTE(sampling_strategy = {"functional needs repair": 5000, "non functional": 22500}, random_state=123456789, n_jobs=8, k_neighbors=7).fit_resample(X_train,y_train)
    print("Numero de instancias: " + str(len(X_train)))
    print("Instancias por clase:")
    print(np.unique(y_train,return_counts=True))


    '''
    print("ADASYN...")
    X_train,y_train = ADASYN(sampling_strategy = {"functional needs repair": 5000, "non functional": 22500}, random_state=123456789, n_jobs=8, n_neighbors=7).fit_resample(X_train,y_train)
    print("Numero de instancias: " + str(len(X_train)))
    print("Instancias por clase:")
    print(np.unique(y_train,return_counts=True))
    '''

    print("Cleaning anomalies...")
    ind_functional = np.where(y_train=="functional")[0]
    ind_non_functional = np.where(y_train=="non functional")[0]
    ind_functional_repair = np.where(y_train=="functional needs repair")[0]
    X1, y1 = cleanAnomalies(X_train[ind_functional], y_train[ind_functional])
    X2, y2 = cleanAnomalies(X_train[ind_non_functional], y_train[ind_non_functional])
    X3, y3 = cleanAnomalies(X_train[ind_functional_repair], y_train[ind_functional_repair])
    X_train = np.concatenate((X1,X2), axis=0)
    X_train = np.concatenate((X_train,X3), axis=0)
    y_train = np.concatenate((y1,y2), axis=0)
    y_train = np.concatenate((y_train,y3), axis=0)
    print("Instancias por clase:")
    print(np.unique(y_train,return_counts=True))


    '''
    print("EditedNearestNeighbours...")
    X_train, y_train = EditedNearestNeighbours(sampling_strategy="all", n_neighbors=7, n_jobs=8).fit_resample(X_train, y_train)
    print("Numero de instancias: " + str(len(X_train)))
    print("Instancias por clase:")
    print(np.unique(y_train,return_counts=True))
    '''

    '''
    print("SSMA...")
    selector = SSMA(n_neighbors=1, alpha=1, max_loop=100, initial_density=0.9).fit(X_train,y_train)
    X_train = selector.X_
    y_train = selector.y_
    print("Numero de instancias: " + str(len(X_train)))
    print("Instancias por clase:")
    print(np.unique(y_train,return_counts=True))
    '''

    '''
    print("Generando la métrica con DML...")
    train_set, _, train_labels, _ = train_test_split(X_train, y_train, train_size=0.25, random_state=123456789)
    print("Tamaño del conjunto original: " + str(len(X_train)) + ", tamaño del train: " + str(len(train_set)))
    dml = NCA().fit(train_set, train_labels)
    X_train = dml.transform(X_train)
    X_test = dml.transform(X_test)
    '''

    return X_train, y_train, id_train, X_test, id_test
