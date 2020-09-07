import os
import numpy as np

def read_files(dir):
    X = []
    y = []
    
    for file in sorted(os.listdir(dir)):
        with open(dir + file, 'r') as f:
            current_file = []
            line = f.readline()
            while line:
                current_file.append([int(x) for x in line.split()])
                line = f.readline()
            
            y.append(current_file.pop(0)[0])
            
            aux = []
            for i in range(len(current_file)):
                for j in range(len(current_file[i])):
                    aux.append(current_file[i][j])
            X.append(aux)

    return X, y

def update_weight(weight, learning_rate, desired_output, adaline_output, x):
    return weight + learning_rate * (desired_output - adaline_output) * x

def train(X_train, y_train, learning_rate):
    weights = np.random.rand(len(X_train[0]))
    max_iter = 10000

    for i in range(max_iter):
        for j in range(len(X_train)):
            sum = 0
            for k in range(len(X_train[j])):
                sum += X_train[j][k] * weights[k]
            
            output = 1 if sum >= 0 else -1    
            if (output != y_train[j]):
                for k in range(len(weights)):
                    weights[k] = update_weight(weights[k], learning_rate, y_train[j], output, X_train[j][k])
                break

    return weights

def test(X_test, y_test, weights):
    for i in range(len(X_test)):
        sum = 0
        for j in range(len(X_test[i])):
            sum += X_test[i][j] * weights[j]

        output = 1 if sum >= 0 else -1
        print("Esperado ", y_test[i], " - Saída do Adaline ", output)

def main():
    X_train, y_train = read_files("train/")
    X_test, y_test = read_files("test/")
    
    learning_rate = 0.000001
    weights = train(X_train, y_train, learning_rate)
    test(X_test, y_test, weights)

if __name__ == '__main__':
    main()