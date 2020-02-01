import numpy as np
import pandas as pd
import time

from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

import pipeline
import preprocessing

np.random.seed(123456789)

classifier = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)

print("Reading train/test sets...")
X_train, y_train, X_test = preprocessing.readData()

print("Pipeline for train set...")
X_train, y_train, _ = pipeline.Pipeline(X_train, y_train)
print("Pipeline for test set...")
X_test,_ , id = pipeline.Pipeline(X_test,train=False, dim=len(X_train[0]))
id = id.astype(int)

print("Fitting KNN classifier...")
classifier.fit(X_train, y_train)

print("Predicting over test set...")
pred = classifier.predict(X_test)
res = pd.DataFrame({"id":id, "status_group":pred})
res.to_csv("../submissions/new_submission.csv", index=False)
