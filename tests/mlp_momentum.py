'''
    Projeto 1 - MLP
    Redes Neurais SCC5809 - ICMC-USP 2020

    Alunos:     Nícolas Roque dos Santos
                Tales Somensi
'''
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer, MinMaxScaler

def read_data(task):
    X = []  # features
    y = []  # labels

    if task == 'r':
        print("Lendo arquivo para regressão.")
        with open("default_features_1059_tracks.txt", 'r') as f:
            for line in f.readlines():
                row = line.strip('\n').split(',')
                X.append(row[:len(row) - 2])
                y.append(row[-2:])
        X = np.array(X, dtype = np.float32)
        y = np.array(y, dtype = np.float32)

    elif task == 'c':
        print("Lendo arquivo para classificação.")
        with open("wine.data", 'r') as f:
            for line in f.readlines():
                row = line.strip('\n').split(',')
                X.append(row[1:])
                y.append(row[0])
        X = np.array(X, dtype = np.float32)
        y = np.array(y, dtype = np.int8)

    else:
        sys.exit("Tarefa inválida.\nPor favor, informe a tarefa corretamente (r, para regressão ou c, para classificação).")

    return X, y

# Cálculo da acurácia do classificador.
def accuracy_score(y_true, y_pred):
    hits = np.sum([np.argmax(y_true[i]) == np.argmax(y_pred[i]) for i in range(len(y_true))])
    return hits / len(y_true)

# Função sigmoid.
def sigmoid(activation):
	return 1.0 / (1.0 + np.exp(-activation))

# Converte y para vetor one-hot.
def to_onehot(y, n):
    onehot = np.zeros([len(y), n])
    for i in range(len(y)):
        onehot[i, y[i] - 1] = 1.0

    return onehot

# Gera os pesos da rede.
def get_weights(layers):
    network = []
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
def transfer_derivative(output):
	return output * (1.0 - output)

# Etapa de forward do backpropagation.
def forward_propagate(X, network):
    y_pred = []
    y_pred.append(X)

    # para cada camada insere o bias, multiplica os peso pelos elementos
    # e aplica a função sigmoid
    for i in range(len(network)):
        # X = np.append(X, [+1])
        X = np.concatenate([X, [+1]])
        #print(X.shape, network[i].shape)
        predicted = sigmoid(np.dot(X, network[i]))
        y_pred.append(predicted)
        X = predicted

    return y_pred

#def update_weight(l_rate, error, y):
#    return l_rate * error * y

def backward_propagate_error(X, y_pred, y_true, network, l_rate, alphaM, previous_delta):

    # print('\n\nnew backpropagate\n\n')
    current_delta = []

    for i in reversed(range(len(network))):         # for each layer (from last hidden to input layer)
        #                                           # e.g.: if there are 3 layers, 'layer' will run as 1 an then 0

        y_bias = np.concatenate([y_pred[i], [+1]])
        delta = []

        # delta_previous = [0] * len(network[i][0])
        # print('delta_previous:', delta_previous)

        for j in range(len(network[i][0])):         # for each neuron on current layer

            if i != len(network) - 1:
                error = np.sum(network[i + 1][j] * delta_old)
            else:
                error = y_true[j] - y_pred[len(network)][j]

            derivative = transfer_derivative(y_pred[i + 1][j])
            delta.append(error * derivative)

            # print('[i][j]: [%d][%d]' %(i, j))

            # print('len(delta):', len(delta))
            # print('delta:', delta)

            for k in range(network[i].shape[0]):    # for each weight on current neuron
                network[i][k, j] += l_rate * delta[j] * y_bias[k] + alphaM * previous_delta[i][j]

        delta_old = delta
        # print('delta_old', delta_old)
        current_delta.append(delta)
        # print('current_delta', current_delta)

    # print('current_delta', current_delta)
    # print('previous_delta', previous_delta)

    curr_delta = []
    for i in reversed(range(len(current_delta))):
        curr_delta.append( current_delta[i] )

    return network, curr_delta

def individual_mse(y_true, y_pred):
    return 1/2 * np.sum([(y_true[i] - y_pred[i]) ** 2 for i in range(len(y_pred))])

def mse(X, y, y_pred):
    error = []
    for i in range(len(X)):
        error.append(individual_mse(y[i], y_pred[i]))

    #error = np.array(error)
    return np.mean(error)

def predict(X, y, network):
    y_pred = []
    for i in range(len(X)):
        predicted = forward_propagate(X[i], network)
        y_pred.append(predicted[len(network)])

    return y_pred

# Treinamento da rede.
def fit(X, y, network, l_rate, epochs, alphaM, epsilon):

    deltaM = [[ 0 for i in range(len(network[j][0])) ] for j in range(len(network)) ]
    # print('len(deltaM)', len(deltaM))
    # print('len(deltaM[0])', len(deltaM[0]))
    # print('len(deltaM[1])', len(deltaM[1]))
    # print('deltaM\n', deltaM)

    for i in range(epochs):
        for j in range(len(X)):
            new_input = X[j]
            y_pred = forward_propagate(new_input, network)
            network, deltaM = backward_propagate_error(X[j], y_pred, y[j], network, l_rate, alphaM, deltaM)

        predicted = predict(X, y, network)
        error = mse(X, y, predicted)

        if not i % 100:
            print( 'Epoch: %d, mse: %f' %( i, error ) )

        if error <= epsilon:
            break

    return network

# Avalia a rede de acordo com a tarefa.
def exec_algorithm(task, X, y, hidden, neurons_hidden, l_rate, epochs, alphaM, epsilon):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, shuffle=True)
    network = initialize_network(hidden, neurons_hidden, len(y_train[0]), len(X_train[0]))
    # print('len(network)      :', len(network))
    # print('len(network[0])   :', len(network[0]))
    # print('len(network[0][0]):', len(network[0][0]))
    # print('len(network[1])   :', len(network[1]))
    # print('len(network[1][0]):', len(network[1][0]))
    network = fit(X_train, y_train, network, l_rate, epochs, alphaM, epsilon)

    if task == 'c':
        predictions = predict(X_test, y_test, network)
        score = accuracy_score(y_test, predictions)
        print("Accuracy: %0.2f " % score)
    elif task == 'r':
        print("todo")
    else:
        sys.exit("Tarefa inválida.\nPor favor, informe a tarefa corretamente (r, para regressão ou c, para classificação).")

def main():
    # Leitura do arquivo de entrada e conversão para one-hot.
    task = 'c' # Usar 'r' para regressão ou 'c' para classificação.
    X, y = read_data(task)
    y = to_onehot(y, 3)

    # Normalização das features.
    transformer = Normalizer().fit(X)
    normalized_X = transformer.transform(X)

    # print('X:\n\n', X)
    # print('\n\nNormalized X:\n\n', normalized_X)
    #
    # sys.exit("\nTest done")

    # Hiperparâmetros.
    epsilon = 0.1
    hidden = 1
    neurons_hidden = 13
    l_rate = 0.2
    epochs = 10000
    alphaM = 0.5

    # Avaliação da rede de acordo com a tarefa.
    exec_algorithm(task, normalized_X, y, hidden, neurons_hidden, l_rate, epochs, alphaM, epsilon)

if __name__ == '__main__':
    main()
