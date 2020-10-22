'''
    Projeto 1 - MLP
    Redes Neurais SCC5809 - ICMC-USP 2020

    Alunos:     Nícolas Roque dos Santos
                Tales Somensi
'''
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer, MinMaxScaler
import matplotlib.pyplot as plt

def read_data():
    X = []  # features
    y = []  # labels
    print("\nLendo arquivo para classificação\n")
    with open("iris.data", 'r') as f:
        for line in f.readlines():
            row = line.strip('\n').split(',')
            X.append(row[:len(row) - 1])

            if(row[-1] == 'Iris-setosa'):
                y.append([1, 0, 0])
            elif(row[-1] == 'Iris-versicolor'):
                y.append([0, 1, 0])
            elif(row[-1] == 'Iris-virginica'):
                y.append([0, 0, 1])
    X = np.array(X, dtype = np.float32)
    y = np.array(y, dtype = np.int8)

    return X, y

# Cálculo da acurácia do classificador.
def accuracy_score(y_true, y_pred):
    hits = np.sum([np.argmax(y_true[i]) == np.argmax(y_pred[i]) for i in range(len(y_true))])
    return hits / len(y_true)

# Função sigmoid.
def sigmoid(activation):
	return 1.0 / (1.0 + np.exp(-activation))

def tanh(activation):
    return np.tanh(activation)

# Gera os pesos da rede.
def get_weights(layers):
    network = list()
    for i in range(len(layers) - 1):
        weight = np.random.normal(loc = 0, scale = 0.2, size = (layers[i] + 1, layers[i + 1])) # bias
        network.append(weight)

    return network

# Inicializa o peso da rede de acordo com o número de camadas (1 ou 2).
def initialize_network(hidden, neurons_hidden, n_classes, n_features):
    if hidden == 1:
        return get_weights([n_features, neurons_hidden, n_classes])
    elif hidden == 2:
        return get_weights([n_features, neurons_hidden, neurons_hidden, n_classes])
    else:
        sys.exit("Número de camadas incorreto (valores possíveis = 1 ou 2).")

# Calcula a derivada da saída de um neuron.
def transfer_derivative_sigmoid(output):
	return output * (1.0 - output)

# Calcula a derivada da saída de um neuron.
def transfer_derivative_tanh(output):
	return ( 1.0 - ( np.tanh(output) ** 2 ) )

# Etapa de forward do backpropagation.
def forward_propagate(X, network, activ_f):
    y_pred = []
    y_pred.append(X)

    # para cada camada insere o bias, multiplica os peso pelos elementos
    # e aplica a função sigmoid
    for i in range(len(network)):
        # X = np.append(X, [+1])
        X = np.concatenate([X, [+1]])
        #print(X.shape, network[i].shape)
        if activ_f == 'sigmoid':
            predicted = sigmoid(np.dot(X, network[i]))
        elif activ_f == 'tanh':
            predicted = tanh(np.dot(X, network[i]))
        y_pred.append(predicted)
        X = predicted

    return y_pred

def backward_propagate_error(X, y_pred, y_true, network, l_rate, activ_f):
    for i in reversed(range(len(network))):
        delta = []
        y_bias = np.concatenate([y_pred[i], [+1]])
        for j in range(len(network[i][0])):
            if i != len(network) - 1:
                error = np.sum(network[i + 1][j] * old_delta)
            else:
                error = y_true[j] - y_pred[len(network)][j]

            if activ_f == 'sigmoid':
                derivative = transfer_derivative_sigmoid(y_pred[i + 1][j])
            elif activ_f == 'tanh':
                derivative = transfer_derivative_tanh(y_pred[i + 1][j])
            delta.append(error * derivative)

            for k in range(network[i].shape[0]):
                delta_w = l_rate * delta[j] * y_bias[k]
                network[i][k, j] += delta_w

        old_delta = delta

    return network

def individual_mse(y_true, y_pred):
    # return 1/2 * np.sum([(y_true[i] - y_pred[i]) ** 2 for i in range(len(y_pred))])
    return 1/2 * np.sum([(y_true[i] - y_pred[i]) ** 2 for i in range(len(y_pred))])

def mse(X, y, y_pred):
    error = []
    for i in range(len(X)):
        error.append(individual_mse(y[i], y_pred[i]))

    #error = np.array(error)
    return np.mean(error)

def predict(X, network, activ_f):
    y_pred = []
    for i in range(len(X)):
        predicted = forward_propagate(X[i], network, activ_f)
        y_pred.append(predicted[len(network)])

    return y_pred

# Treinamento da rede.
def fit(X, y, network, l_rate, epochs, epsilon, activ_f):
    scores = list()
    interrupted = 0
    for i in range(epochs):
        for j in range(len(X)):
            new_input = X[j]
            y_pred = forward_propagate(new_input, network, activ_f)
            network = backward_propagate_error(X[j], y_pred, y[j], network, l_rate, activ_f)

        predicted = predict(X, network, activ_f)
        # error = mse(X, y, predicted)
        error = mse(X, y, predicted)

        scores.append(accuracy_score(y, predicted))
        # scores = np.array(scores, dtype = np.float32)

        if not i % 10:
            print( 'Epoch: %d, mse: %f' %( i, error ) )

        if error <= epsilon:
            # print("l_rate %.1f\nTrain accuracy: %.2f " % (l_rate, scores.mean()))
            print("\nl_rate %.1f\nTrain accuracy: %.2f " % (l_rate, np.mean(scores)))
            print("error na saida %d: %.2f" % (i, error))
            interrupted = 1
            break

    if(interrupted == 0):
        print("l_rate %.1f\nTrain accuracy: %.2f " % (l_rate, np.mean(scores)))

    return network, predicted

# Avalia a rede de acordo com a tarefa.
def exec_algorithm(X, y, hidden, neurons_hidden, l_rate, epochs, epsilon, activ_f):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, shuffle = True)
    start = time.time()
    network = initialize_network(hidden, neurons_hidden, len(y_train[0]), len(X_train[0]))
    network, predicted = fit(X_train, y_train, network, l_rate, epochs, epsilon, activ_f)
    end = time.time()
    print("Tempo de execução: %.2f segundos" % (end - start))

    predictions = predict(X_test, network, activ_f)

    score = accuracy_score(y_test, predictions)
    print("Test accuracy: %0.2f " % score)
    plt.plot(predicted)
    plt.show()

def main():
    # Leitura do arquivo de entrada e conversão para one-hot.
    X, y = read_data()
    # Normalização das features.
    transformer = Normalizer().fit(X)
    normalized_X = transformer.transform(X)

    # Hiperparâmetros.
    epsilon = 0.05
    hidden = 1
    neurons_hidden = 2
    epochs = 10000
    l_rate = 0.5

    print('\nrunning for sigmoid:\n')
    activ_f = 'sigmoid'
    exec_algorithm(normalized_X, y, hidden, neurons_hidden, l_rate, epochs, epsilon, activ_f)
    print('\nrunning for tanH:\n')
    activ_f = 'tanh'
    exec_algorithm(normalized_X, y, hidden, neurons_hidden, l_rate, epochs, epsilon, activ_f)

if __name__ == '__main__':
    main()
