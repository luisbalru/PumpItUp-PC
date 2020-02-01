import preprocessing
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from sklearn import preprocessing

import numpy as np

def Pipeline(X,y=[], train=True, dim=-1):
    id = np.array(X["id"])
    X = X.drop(columns=["id"])
    X = np.array(X)
    y = np.array(y)

    X = preprocessing.scale(X)

    '''
    if dim==-1:
        X = PCA().fit_transform(X)
    else:
        X = PCA(n_components=dim).fit_transform(X)

    if train:
        X,y = SMOTE(random_state=123456789).fit_resample(X,y)
    '''
    return X,y,id
