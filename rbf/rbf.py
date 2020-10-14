'''
    Exercicio 3 - RBF e MLP
    Redes Neurais SCC5809 - ICMC-USP 2020
    
    Alunos:     Nícolas Roque dos Santos
                Tales Somensi
'''

import sys
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold

def read_data(filename):
    #
    # import 'wine.data'
    # and split it into X (features) and y (labels)
    #
    X = []  # features
    y = []  # labels
    with open(filename, 'r') as f:
        for line in f.readlines():
            row = line.strip('\n').split(',')
            X.append(row[1:])   # appends elements from second column to the last one
            y.append(row[0])    # appends only first column element
    X = np.array(X, dtype = np.float32)
    y = np.array(y, dtype = np.int8)
    
    return X, y

def get_centers(X, n_clusters, init, max_iter):
    kmeans = KMeans(n_clusters = n_clusters, init = init, max_iter = max_iter).fit(X)
    return kmeans.cluster_centers_

def get_widths(centers, m):
    sum = 0
    for i in range(m):
        smallest_distance = sys.float_info.max
        for j in range(m):        
            if i != j:
                distance = np.linalg.norm(centers[i] - centers[j])
                if distance < smallest_distance:
                    smallest_distance = distance
        sum += smallest_distance
    
    return [sum / m] * m

def rbf(x, center, width):
    v = np.linalg.norm(x - center)
    return np.exp(-1 * (v ** 2) / (2 * width ** 2))

def update_weight(weight, learning_rate, desired_output, rbf_output, phi_x):
    return weight + learning_rate * (desired_output - rbf_output) * phi_x

def accuracy_score(y_true, y_pred):
    hits = np.sum([y_true[i] == y_pred[i] for i in range(len(y_true))])
    return hits / len(y_true)

def fit(X, y, learning_rate, centers, widths, n_bases, epochs):
    w = np.random.rand(n_bases)
    epochs = epochs

    for i in range(epochs):
        for j in range(len(X)):
            sum = 0
            phi = np.array([rbf(X[j], center, width) for center, width in zip(centers, widths)])
            for k in range(n_bases):
                sum += w[k] * phi[k]

            for k in range(n_bases):
                w[k] = update_weight(w[k], learning_rate, y[j], sum, phi[k]) 
    
    return w

def predict(X, w, learning_rate, centers, widths, n_bases):
    y_pred = []
    for i in range(len(X)):
        sum = 0
        phi = np.array([rbf(X[i], center, width) for center, width in zip(centers, widths)])
        for j in range(n_bases):
            sum += w[j] * phi[j]
        y_pred.append(int(round(sum)))

    return y_pred

def main():
    filename = "wine.data"
    X, y = read_data(filename)
    
    n_bases = 4
    learning_rate = 0.01
    centers = get_centers(X, n_bases, "k-means++", 100)
    widths = get_widths(centers, n_bases)
    
    epochs = 100
    scores = []
    kf = KFold(n_splits = 10, shuffle = False)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        w = fit(X_train, y_train, learning_rate, centers, widths, n_bases, epochs)
        y_pred = predict(X_test, w, learning_rate, centers, widths, n_bases)
        scores.append(accuracy_score(y_test, y_pred))
    
    scores = np.array(scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

if __name__ == '__main__':
    main()