from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.under_sampling import RandomUnderSampler
import pandas as pd
import numpy as np

def range01(x):
    n =((x-np.max(x))/(np.max(x)-np.min(x)))
    return(n)

def aplicaSMOTE(X_train, y_train):
    X_train, y_train = SMOTE(random_state=77145416,n_jobs=14).fit_resample(X_train, y_train)
    print("Numero de instancias: " + str(len(X_train)))
    print("Instancias por clase:")
    print(np.unique(y_train,return_counts=True))
    df = pd.concat([X_train.reset_index(drop=True), y_train], axis = 1)
    return(df)

def aplicaENN(X_train, y_train):
    X_train, y_train = EditedNearestNeighbours(n_neighbors=7, n_jobs=14).fit_resample(X_train, y_train)
    print("Numero de instancias: " + str(len(X_train)))
    print("Instancias por clase:")
    print(np.unique(y_train,return_counts=True))
    df = pd.concat([X_train.reset_index(drop=True), y_train], axis = 1)
    return(df)

def aplicaRUS(X_train,y_train):
    X_train, y_train = RandomUnderSampler(random_state = 77145416).fit_resample(X_train, y_train)
    print("Numero de instancias: " + str(len(X_train)))
    print("Instancias por clase:")
    print(np.unique(y_train,return_counts=True))
    df = pd.concat([X_train.reset_index(drop=True), y_train], axis = 1)
    return(df)

def toCSV(df, nombre_archivo):
    df.to_csv(nombre_archivo, index = False)

datos_rus = pd.read_csv("para_rus.csv")
df_rus = aplicaRUS(datos_rus.drop(columns='status_group'),datos_rus['status_group'])
df_rus.iloc[:,0:7] = range01(df_rus.iloc[:,0:7])
toCSV(df_rus.drop(columns='class'),'tras_rus.csv')

#datos1 = pd.read_csv("rovun_dummies.csv")
#df = aplicaENN(datos1.drop(columns='status_group'),datos1['status_group'])
#toCSV(df,"dataset-enn.csv")