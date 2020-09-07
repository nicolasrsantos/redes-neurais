import os
import numpy as np

def read_files(dir):
    files = []
    
    for file in os.listdir(dir):
        with open(dir + file, 'r') as f:
            current_file = []
            line = f.readline()
            
            while line:
                current_file.append(line.split())
                line = f.readline()
            files.append(current_file)
    
    return files

def update_weight(weight, learning_rate, desired_output, calculated_output, x):
    return weight + learning_rate * (desired_output - calculated_output) * x

def adaline(train, test, learning_rate):
    

def main():
    train = read_files("train/")
    test = read_files("test/")

    learning_rate = 0.5
    adaline(train, test, learning_rate)

if __name__ == '__main__':
    main()