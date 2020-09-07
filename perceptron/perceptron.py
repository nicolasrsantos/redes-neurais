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

def print_letter(letter):
    for i in range(5):
        print(' '.join(letter[i]))
        

def main():
    train = read_files("train/")
    test = read_files("test/")
    
    

if __name__ == '__main__':
    main()