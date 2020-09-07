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
                current_file.append(line.split())
                line = f.readline()
            
            y.append(current_file.pop(0))
            X.append(current_file)
    
    return X, y

def update_weight(weight, learning_rate, desired_output, adaline_output, x):
    return weight + learning_rate * (desired_output - adaline_output) * x

def adaline_train(X_train, y_train, learning_rate):

def adaline_test(X_test, y_test):

def main():
    X_train, y_train = read_files("train/")
    X_test, y_test = read_files("test/")
    
    for i in range(6):
        print(y_test[i])
        for j in range(5):
            print(X_test[i][j])
        print("")
    
    learning_rate = 0.5
    adaline_train(X_train, y_train, learning_rate)
    adaline_test(X_test, y_test)

if __name__ == '__main__':
    main()