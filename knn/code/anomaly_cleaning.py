from pyod.models.auto_encoder import AutoEncoder
import numpy as np

def cleanAnomalies(X,y,perc=0.07):
    detector = AutoEncoder(hidden_neurons=[256,128,64,32,16,16,32,64,128,256], epochs=50).fit(X)
    sorted = np.argsort(detector.decision_scores_)[::-1]
    size = len(X)-int(perc*len(X))
    return X[sorted[:size]], y[sorted[:size]]
