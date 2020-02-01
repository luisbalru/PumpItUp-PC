import numpy as np
import pandas as pd
import time

from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

import pipeline
import preprocessing

np.random.seed(123456789)

classifier = KNeighborsClassifier(n_neighbors=7, n_jobs=-1)

print("Reading train/test sets...")
X_train, y_train, X_test = preprocessing.readData()

print("Pipeline for train/test sets...")
X_train, y_train, id_train, X_test, id_test = pipeline.Pipeline(X_train, y_train, X_test)
id_test = id_test.astype(int)

print("Fitting KNN classifier...")
classifier.fit(X_train, y_train)

print("Predicting over test set...")
pred = classifier.predict(X_test)
res = pd.DataFrame({"id":id_test, "status_group":pred})
res.to_csv("../submissions/new_submission.csv", index=False)
