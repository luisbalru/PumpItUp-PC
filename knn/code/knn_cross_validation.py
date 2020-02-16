import numpy as np
import pandas as pd
import time

from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

import pipeline
import preprocessing

np.random.seed(123456789)

def F1_score(conf_matrix):
    """Calculates the F1 score

    Parameters:
        conf_matrix (numpy ndarray): Confusion matrix of the experiment

    Returns:
        f1_score (double): The calculated F1 score
    """
    row_sums = np.sum(conf_matrix, axis=1)
    col_sums = np.sum(conf_matrix, axis=0)
    diag = np.diag(conf_matrix)
    scores = 2*diag / (row_sums + col_sums)
    return np.mean(scores)

accs = []
f1s = []
tiempos = []

#possibleK = [1,3,5,7,9,11]
#dims = list(range(30,50))
#possibleK=[1]*len(dims) + [3]*len(dims) + [5]*len(dims) + [7]*len(dims)
possibleK=[1]
dims=[44]
X,y, _ = preprocessing.readData()
print("El conjunto de datos tiene dimension: " + str(len(X.iloc[0])))

for k,dim in zip(possibleK, dims):
    print("K="+str(k))
    print("Dim="+str(dim))
    classifier = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    skf = StratifiedKFold(n_splits=5, random_state=123456789)

    scores = []
    f1_scores = []
    elapsed_times = []
    conf_matrix = []

    X_index = skf.split(X,y)
    missclassified_images = []

    for train_index, test_index in X_index:
        print("New split")
        print("Train size: {}, test size: {}".format(len(train_index), len(test_index)))

        train_set, train_labels = X.iloc[train_index], y[train_index]
        test_set, test_labels = X.iloc[test_index], y[test_index]

        train_set,train_labels,id_train,test_set,id_test = pipeline.Pipeline(train_set, train_labels, test_set, dim)

        start_time = time.time()
        classifier.fit(train_set, train_labels)
        labels_pred = classifier.predict(test_set)
        finish_time = time.time()

        fails = pd.DataFrame(
            {
                'preds': labels_pred,
                'reals': test_labels
            },
            index = test_index
        )

        fails = fails[fails['preds'] != fails['reals']]
        missclassified_images.append(fails)
        print("The number of fails for this split is ", len(fails))
        print(fails)

        curr_score = metrics.accuracy_score(labels_pred, test_labels)
        curr_mat = metrics.confusion_matrix(test_labels, labels_pred)
        curr_f1 = F1_score(curr_mat)

        elapsed_times.append(finish_time - start_time)
        conf_matrix.append(curr_mat)
        scores.append(curr_score)
        f1_scores.append(curr_f1)

    print("Matrices de confusion")
    for mat in conf_matrix:
        print(mat)

    results = pd.DataFrame(
        {
            'Precision (%)': np.asarray(scores)*100,
            'F1 (%)': np.asarray(f1_scores)*100,
            'Tiempo (s)': elapsed_times,
        }
    )

    results.index += 1

    results.loc['Total'] = results.mean()
    latex_list = results.to_latex(float_format='%.3f').replace('lrrr', 'lccc').splitlines()
    latex_list.insert(len(latex_list)-3, '\midrule')
    latex_new = '\n'.join(latex_list)

    accs.append(np.mean(np.asarray(scores))*100)
    f1s.append(np.mean(np.asarray(f1_scores))*100)
    tiempos.append(np.mean(np.asarray(elapsed_times)))

    print(latex_new)

mejor = [1,0,-1,-1,-1]
for k,d,a,f1,t in zip(possibleK,dims,accs,f1s,tiempos):
    print("\nPara k=" + str(k) + ", ndims=" + str(d) + " los resultados son:")
    print("Accuracy medio: " + str(a) + "%")
    print("F1 medio: " + str(f1) + "%")
    print("Tiempo medio: " + str(t) + "s")
    if(mejor[1]<a):
        mejor[0]=k
        mejor[1]=a
        mejor[2]=f1
        mejor[3]=t
        mejor[4]=d

print("\n\n\nEl mejor resultado obtenido es con k=" + str(mejor[0]) + ", ndims=" + str(mejor[4]) + " con resultados:")
print("Accuracy medio: " + str(mejor[1]) + "%")
print("F1 medio: " + str(mejor[2]) + "%")
print("Tiempo medio: " + str(mejor[3]) + "s")
