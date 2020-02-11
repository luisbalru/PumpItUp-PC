# -*- coding: utf-8 -*-
"""
Steady-State Memetic Algorithm
"""

# Author: Dayvid Victor <victor.dvro@gmail.com>
#
# License: BSD 3 clause

import random
import numpy as np


from sklearn.utils.validation import check_X_y
from sklearn.neighbors.classification import KNeighborsClassifier

from sklearn.metrics import accuracy_score

from base import InstanceReductionMixin

from sklearn.model_selection import StratifiedKFold


class SSMA(InstanceReductionMixin):
    """Steady State Memetic Algorithm

    The Steady-State Memetic Algorithm is an evolutionary prototype
    selection algorithm. It uses a memetic algorithm in order to
    perform a local search in the code.

    Parameters
    ----------
    n_neighbors : int, optional (default = 3)
        Number of neighbors to use by default for :meth:`k_neighbors` queries.

    alpha   : float (default = 0.6)
        Parameter that ponderates the fitness function.

    max_loop    : int (default = 1000)
        Number of maximum loops performed by the algorithm.

    threshold   : int (default = 0)
        Threshold that regulates the substitution condition;

    chromosomes_count: int (default = 10)
        number of chromosomes used to find the optimal solution.

    Attributes
    ----------
    `X_` : array-like, shape = [indeterminated, n_features]
        Selected prototypes.

    `y_` : array-like, shape = [indeterminated]
        Labels of the selected prototypes.

    `reduction_` : float, percentual of reduction.

    Examples
    --------
    >>> from protopy.selection.ssma import SSMA
    >>> import numpy as np
    >>> X = np.array([[i] for i in range(100)])
    >>> y = np.asarray(50 * [0] + 50 * [1])
    >>> ssma = SSMA()
    >>> ssma.fit(X, y)
    SSMA(alpha=0.6, chromosomes_count=10, max_loop=1000, threshold=0)
    >>> print ssma.predict([[40],[60]])
    [0 1]
    >>> print ssma.reduction_
    0.98

    See also
    --------
    sklearn.neighbors.KNeighborsClassifier: nearest neighbors classifier

    References
    ----------
    Joaquín Derrac, Salvador García, and Francisco Herrera. Stratified prototype
    selection based on a steady-state memetic algorithm: a study of scalability.
    Memetic Computing, 2(3):183–199, 2010.

    """
    def __init__(self, n_neighbors=1, alpha=0.6, max_loop=1000, threshold=0, chromosomes_count=10, initial_density=0.9):
        self.n_neighbors = n_neighbors
        self.alpha = alpha
        self.max_loop = max_loop
        self.threshold = threshold
        self.chromosomes_count = chromosomes_count

        self.evaluations = None
        self.chromosomes = None

        self.best_chromosome_ac = -1
        self.best_chromosome_rd = -1

        self.initial_density = initial_density

        self.classifier = KNeighborsClassifier(n_neighbors = n_neighbors, n_jobs=-1)


    def accuracy(self, chromosome, X, y):
        mask = np.asarray(chromosome, dtype=bool)
        cX, cy = X[mask], y[mask]

        '''
        skf = StratifiedKFold(n_splits=5)
        splits_indices = skf.split(cX, cy)

        scores = []

        for train_index, test_index in splits_indices:
            self.classifier.fit(cX[train_index], cy[train_index])
            labels = self.classifier.predict(cX[test_index])
            scores.append(accuracy_score(cy[test_index],labels))

        scores = np.array(scores)

        return np.mean(scores)
        '''
        self.classifier.fit(cX,cy)
        labels = self.classifier.predict(X)
        return accuracy_score(y, labels)


    def fitness(self, chromosome, X, y):
        #TODO add the possibility of use AUC for factor1
        ac = self.accuracy(chromosome, X, y)
        rd = 1.0 - (float(np.sum(chromosome))/len(chromosome))

        return self.alpha * ac + (1.0 - self.alpha) * rd


    def fitness_gain(self, gain, n):
        return self.alpha * (float(gain)/n) + (1 - self.alpha) * (1.0 / n)


    def update_threshold(self, X, y):
        best_index = np.argmax(self.evaluations)
        chromosome = self.chromosomes[best_index]

        best_ac = self.accuracy(chromosome, X, y)
        best_rd = 1.0 - float(sum(chromosome))/len(y)

        if best_ac <= self.best_chromosome_ac:
            self.threshold = self.threshold + 1
        if best_rd <= self.best_chromosome_rd:
            self.threshold = self.threshold - 1

        self.best_chromosome_ac = best_ac
        self.best_chromosome_rd = best_rd


    def index_nearest_neighbor(self, S, X, y):
        classifier = KNeighborsClassifier(n_neighbors=1)

        U = []
        S_mask = np.array(S, dtype=bool, copy=True)
        indexs = np.asarray(range(len(y)))[S_mask]
        X_tra, y_tra = X[S_mask], y[S_mask]

        for i in range(len(y)):
            real_indexes = np.asarray(range(len(y)))[S_mask]
            X_tra, y_tra = X[S_mask], y[S_mask]
            #print len(X_tra), len(y_tra)
            classifier.fit(X_tra, y_tra)
            [[index]] = classifier.kneighbors(X[i].reshape(1,-1), return_distance=False)
            U = U + [real_indexes[index]]

        return U


    def memetic_looper(self, S, R):
        c = 0
        for i in range(len(S)):
            if S[i] == 1 and i not in R:
                c = c + 1
                if c == 2:
                    return True

        return False

    def memetic_select_j(self, S, R):
        indexs = []
        for i in range(len(S)):
            if i not in R and S[i] == 1:
                indexs.append(i)
        # if list is empty wlil return error
        return np.random.choice(indexs)


    def generate_population(self, X, y):
        ones_density = int(self.initial_density*len(y))
        self.chromosomes = []
        print("Generando poblacion...")
        for i in range(self.chromosomes_count-1):
            ind = np.random.choice(len(X), ones_density, replace=False)
            c = np.zeros(len(X))
            c[ind]=1
            self.chromosomes.append(c)
        print("Evaluando la poblacion...")
        self.chromosomes.append(np.ones(len(X)))
        self.evaluations = [self.fitness(c, X, y) for c in self.chromosomes]

        self.update_threshold(X, y)


    def select_parents(self, X, y):
        parents = []
        for i in range(2):
            samples = random.sample(self.chromosomes, 2)
            parents = parents + [samples[0] if self.fitness(samples[0], X, y) >
                                    self.fitness(samples[1], X, y) else samples[1]]
        return np.array(parents, copy=True)

    def crossover(self, parent_1, parent_2):
        size = len(parent_1)
        mask = [0 if i<int(size/2) else 1 for i in range(size)]
	#mask = [0] * (size/2) + [1] * (size - size/2)
        mask = np.asarray(mask, dtype=bool)
        np.random.shuffle(mask)

        off_1 = parent_1 * mask + parent_2 * ~mask
        off_2 = parent_2 * mask + parent_1 * ~mask

        return np.asarray([off_1, off_2])


    def mutation(self, offspring):
        for i in range(len(offspring)):
            if np.random.uniform(0,1) < 1.0/len(offspring):
                offspring[i] = not offspring[i]

        return offspring

    def memetic_search(self, chromosome, X, y, chromosome_fitness = None):
        S = np.array(chromosome, copy=True)
        if S.sum() == 0:
            return S, 0

        if chromosome_fitness == None:
            chromosome_fitness = self.fitness(chromosome, X, y)
        fitness_s = chromosome_fitness

        # List of visited genes in S
        R = []
        # let U = {u0, u1, ..., un} list where ui = classifier(si,S)/i
        U = self.index_nearest_neighbor(S, X, y)

        while self.memetic_looper(S, R):
            j = self.memetic_select_j(S, R)
            S[j] = 0
            gain = 0.0
            U_copy = list(U)
            mask = np.asarray(S, dtype=bool)
            X_tra, y_tra = X[mask], y[mask]
            real_idx = np.asarray(range(len(y)))[mask]

            if len(y_tra) > 0:
                for i in range(len(U)):
                    if U[i] == j:
                        self.classifier.fit(X_tra, y_tra)
                        [[idx]] = self.classifier.kneighbors(X[i].reshape(1,-1), n_neighbors=1,
                                return_distance=False)
                        U[i] = real_idx[idx]

                        if y[i] == y[U_copy[i]] and y[i] != y[U[i]]:
                            gain = gain - 1.0
                        if y[i] != y[U_copy[i]] and y[i] == y[U[i]]:
                            gain = gain + 1.0

            if gain >= self.threshold:
                n = S.sum()
                g = self.fitness_gain(gain, n)
                fitness_s = fitness_s + g
                R = []
            else:
                U = U_copy
                S[j] = 1
                R.append(j)

        return list(S), fitness_s




    def main_loop(self, X, y):
        print("Generando la población inicial...")
        self.generate_population(X, y)
        n, worse_fit_index = 0, -1
        while (n < self.max_loop):

            print("Iteracion " + str(n) + "/" + str(self.max_loop) + ", mejor rendimiento: " + str(np.amax(self.evaluations)))
            print("Longitud original: " + str(len(y)) + ", longitud actual: " + str(np.sum(self.chromosomes[np.argmax(self.evaluations)])))

            #print("Iteracion " + str(n) + "/" + str(self.max_loop) + ", mejor rendimiento: " + str(np.amax(self.evaluations)))

            parents = self.select_parents(X, y)
            offspring = self.crossover(parents[0], parents[1])
            #print("Offspring antes de la mutacion: " + str(offspring))
            offspring[0] = self.mutation(offspring[0])
            offspring[1] = self.mutation(offspring[1])
            #print("Offspring despues de la mutacion: " + str(offspring))

            fit_offs = [self.fitness(off, X, y) if sum(off) > 0 else -1 for off in offspring]

            if worse_fit_index == -1:
                worse_fit_index = np.argmin(self.evaluations)

            '''
            for i in range(len(offspring)):
                p_ls = 1.0

                if fit_offs[i] == -1:
                    p_ls = -1

                if fit_offs[i] <= self.evaluations[worse_fit_index]:
                    p_ls = 0.0625

                if np.random.uniform(0,1) < p_ls:

                    offspring[i], fit_offs[i] = self.memetic_search(offspring[i], X, y, chromosome_fitness = fit_offs[i])
            '''

            for i in range(len(offspring)):
                if fit_offs[i] > self.evaluations[worse_fit_index] or (fit_offs[i]==self.evaluations[worse_fit_index] and np.sum(offspring[i]<np.sum(self.chromosomes[worse_fit_index]))):
                    #print("Se ha modificado un elemento: " + str(worse_fit_index) + ", fit antiguo: " + str(self.evaluations[worse_fit_index]) + ", fit nuevo: " + str(fit_offs[i]))
                    #print("Antigua longitud: " + str(np.sum(self.chromosomes[worse_fit_index])) + ", nueva longitud: " + str(np.sum(offspring[i])) + ", longitud total: " + str(len(y)))
                    self.chromosomes[worse_fit_index] = offspring[i]
                    self.evaluations[worse_fit_index] = fit_offs[i]

                    worse_fit_index = np.argmin(self.evaluations)

            n = n + 1
            if n % 10 == 0:
                self.update_threshold(X, y)

            #import pdb; pdb.set_trace()


    def reduce_data(self, X, y):
        X, y = check_X_y(X, y, accept_sparse="csr")

        classes = np.unique(y)
        self.classes_ = classes

        self.main_loop(X, y)

        max_value = np.amax(self.evaluations)
        best_index = np.argmax(self.evaluations)
        for i in range(len(self.evaluations)):
            if self.evaluations[i]==max_value and np.sum(self.chromosomes[i])<np.sum(self.chromosomes[best_index]):
                best_index=i
        mask = np.asarray(self.chromosomes[best_index], dtype=bool)
        self.X_ = X[mask]
        self.y_ = y[mask]
        self.reduction_ = 1.0 - float(len(self.y_))/len(y)

        return self.X_, self.y_
