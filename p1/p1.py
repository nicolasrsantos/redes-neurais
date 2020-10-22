'''
    Prova 1
    Redes Neurais SCC5809 - ICMC-USP 2020

    Alunos:     Nícolas Roque dos Santos
                Tales Somensi
'''
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer, MinMaxScaler

def read_data(filename):
    X = []
    y = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            row = line.strip('\n').split(',')
            if row[0] != '':
                X.append(row[:len(row) - 1])
                y.append(row[-1:])
    X = np.array(X, dtype = np.float32)
    y = np.array(y)
    
    return X, y

def rbf(x, center, width):
    v = np.linalg.norm(x - center)
    return np.exp(-1 * (v ** 2) / (2 * width ** 2))

def get_centers(X, n_clusters):
    kmeans = KMeans(n_clusters = n_clusters).fit(X)
    return kmeans.cluster_centers_

def to_onehot(y):
    n_cat = 3
    labels = np.zeros([len(y), n_cat])
    for i in range(len(y)):
        if y[i][0] == 'Iris-virginica':
            labels[i] = [0., 0., 1.]
        elif y[i][0] == 'Iris-versicolor':
            labels[i] = [0., 1., 0.]
        else:
            labels[i] = [1., 0., 0.]

    return labels

def activation(element):
    if element > 0:
        return 1.
    return 0.

def update_weight(weight, eta, desired_output, rbf_output, phi_x):
    return weight + eta * (desired_output - rbf_output) * phi_x

def fit(X, y, eta, centers, widths, n_centroids, epochs):
    w = np.random.rand(n_centroids)
    phi = []
    for i in range(epochs):
        for j in range(len(X)):
            print("novo phi")
            for k in range(n_centroids):
                phi.append(rbf(X[j], centers[k], widths[k]))
            V = np.dot(phi, w)
            Y = [activation(v) for v in V]
            
            for k in range(n_bases):
                w[k] = update_weight(w[k], learning_rate, y[j], sum, Y[k]) 
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
    X, y = read_data('iris.data')
    
    # Normalização das features e one dos labels.
    transformer = Normalizer().fit(X)
    normalized_X = transformer.transform(X)
    y = to_onehot(y)

    epochs = 100
    n_centroids = 3
    eta = 0.01
    centers = get_centers(X, n_centroids)
    widths = [1] * n_centroids

    X_train, X_test, y_train, y_test = train_test_split(normalized_X, y, test_size = 0.33, shuffle = True)
    w = fit(X_train, y_train, eta, centers, widths, n_centroids, epochs)
    y_pred = predict(X_test, w, learning_rate, centers, widths, n_bases)

if __name__ == '__main__':
    main()