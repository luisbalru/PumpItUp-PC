import preprocessing
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import AllKNN
from sklearn import preprocessing
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import plotly.express as px
import plotly

from sklearn.model_selection import train_test_split
#from dml.lmnn import KLMNN
#from dml.nca import NCA
#from dml.lmnn import LMNN

import autoencoder
import autoencoder_denoising

import numpy as np
import pandas as pd

from ipf import IPF

from sklearn.feature_selection import SelectKBest, f_classif

from imblearn.under_sampling import EditedNearestNeighbours

from ssma import SSMA

from anomaly_cleaning import cleanAnomalies

colors = ["red", "blue", "green"]

def plotData(X, y, route):
    reduced = TSNE(n_components=2, n_jobs=-1).fit_transform(X)

    cl0 = np.array([reduced[i] for i in range(len(reduced)) if y[i]=="functional"])
    cl1 = np.array([reduced[i] for i in range(len(reduced)) if y[i]=="functional needs repair"])
    cl2 = np.array([reduced[i] for i in range(len(reduced)) if y[i]=="non functional"])

    print("Número de elementos de la clase 'functional' : " + str(len(cl0)))
    print("Número de elementos de la clase 'functional needs repair' : " + str(len(cl1)))
    print("Número de elementos de la clase 'non functional' : " + str(len(cl2)))

    plt.scatter(cl0[:,0], cl0[:,1], color = colors[0], label = "Functional")
    plt.scatter(cl1[:,0], cl1[:,1], color = colors[1], label = "Functional needs repair")
    plt.scatter(cl2[:,0], cl2[:,1], color = colors[2], label = "Non functional")
    plt.legend()
    plt.savefig(route+"_2d.png")

    reduced = TSNE(n_components=3, n_jobs=-1).fit_transform(X)

    d = pd.DataFrame({"x": reduced[:,0], "y": reduced[:,1], "z": reduced[:,2], "labels": y})
    fig = px.scatter_3d(d, x="x", y="y", z="z", color="labels")
    fig.update_traces(marker=dict(size=5,
                                  line=dict(width=1,
                                            color='DarkSlateGrey')),
                      selector=dict(mode='markers'))
    plotly.offline.plot(fig, filename=route+"_3d.html", auto_open=True)

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

    plotData(X_train, y_train, "raw")

    print("Scaling data...")
    X_train = preprocessing.scale(X_train)
    X_test = preprocessing.scale(X_test)

    plotData(X_train, y_train, "scaled")

    print("PCA con " + str(n_dims) + " componentes...")
    X_train_binary = np.delete(X_train, ind_numeric, axis=1)
    X_test_binary = np.delete(X_test, ind_numeric, axis=1)
    X_train_numeric = X_train[:,ind_numeric]
    X_test_numeric = X_test[:,ind_numeric]
    pca = PCA(n_components=n_dims)
    #pca = KernelPCA(n_components=n_dims, kernel="linear", n_jobs=-1)
    X1 = pca.fit_transform(X_train_binary)
    X2 = pca.transform(X_test_binary)
    X_train = np.hstack((X_train_numeric, X1))
    X_test = np.hstack((X_test_numeric, X2))
    print("Numero de features: " + str(len(X_train[0])))

    plotData(X_train, y_train, "PCA")

    '''
    print("Reduccion de dimensionalidad con AutoEncoder...")
    hid = [50,60,50]
    X_train, X_test = autoencoder.fitTransform(X_train, X_test, 50, hid, bsize=32)
    print("Numero de features: " + str(len(X_train[0])))
    '''

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

    plotData(X_train, y_train, "IPF")

    '''
    print("Denoising autoencoder...")
    hid = [32,16,32]
    X_train, X_test = autoencoder_denoising.fitTransform(X_train, X_test, 250, hid, bsize=32, kreg=None, areg=None)
    '''

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
    X_train,y_train = SMOTE(sampling_strategy = {"functional needs repair": 7500, "non functional": 22000}, random_state=123456789, n_jobs=20, k_neighbors=7).fit_resample(X_train,y_train)
    print("Numero de instancias: " + str(len(X_train)))
    print("Instancias por clase:")
    print(np.unique(y_train,return_counts=True))

    plotData(X_train, y_train, "SMOTE")

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

    plotData(X_train, y_train, "anomalias_knn")

    '''
    print("EditedNearestNeighbours...")
    X_train, y_train = EditedNearestNeighbours(sampling_strategy="not minority", n_neighbors=15, n_jobs=20, kind_sel="mode").fit_resample(X_train, y_train)
    print("Numero de instancias: " + str(len(X_train)))
    print("Instancias por clase:")
    print(np.unique(y_train,return_counts=True))
    '''

    '''
    print("SSMA...")
    selector = SSMA(n_neighbors=1, alpha=0.95, max_loop=10, initial_density=0.9).fit(X_train,y_train)
    X_train = selector.X_
    y_train = selector.y_
    print("Numero de instancias: " + str(len(X_train)))
    print("Instancias por clase:")
    print(np.unique(y_train,return_counts=True))
    '''

    '''
    print("Generando la métrica con DML...")
    train_set, _, train_labels, _ = train_test_split(X_train, y_train, train_size=0.5, random_state=123456789)
    print("Tamaño del conjunto original: " + str(len(X_train)) + ", tamaño del train: " + str(len(train_set)))
    dml = KLMNN().fit(train_set, train_labels)
    X_train = dml.transform(X_train)
    X_test = dml.transform(X_test)
    '''

    return X_train, y_train, id_train, X_test, id_test
