'''
    Projeto 1 - MLP
    Redes Neurais SCC5809 - ICMC-USP 2020

    Alunos:     Nícolas Roque dos Santos
                Tales Somensi
'''
import numpy as np
import random
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer, MinMaxScaler

# Função para leitura dos dados.
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
    network = list()
    previous_delta = list()
    for i in range(len(layers) - 1):
        weight = np.random.normal(loc = 0, scale = 0.2, size = (layers[i] + 1, layers[i + 1])) # bias
        network.append(weight)
        delta = np.zeros((layers[i] + 1, layers[i + 1]), dtype = float)
        previous_delta.append(delta)

    return network, previous_delta

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

# Backpropaga o erro, atualizando os pesos e salvando os deltas da iteração anterior.
def backward_propagate_error(X, y_pred, y_true, network, l_rate, alpha, previous_delta):
    for i in reversed(range(len(network))):
        delta = []
        y_bias = np.concatenate([y_pred[i], [+1]])
        for j in range(len(network[i][0])):
            if i != len(network) - 1:
                error = np.sum(network[i + 1][j] * old_delta)
            else:
                error = y_true[j] - y_pred[len(network)][j]
            
            derivative = transfer_derivative(y_pred[i + 1][j])
            delta.append(error * derivative)
            
            for k in range(network[i].shape[0]):
                delta_w = l_rate * delta[j] * y_bias[k] + alpha * previous_delta[i][k, j]
                network[i][k, j] += delta_w
                previous_delta[i][k, j] = delta_w
        old_delta = delta
        
    return network, previous_delta

# Calcula o erro quadrático médio individual.
def individual_mse(y_true, y_pred):
    return np.sum([(y_true[i] - y_pred[i]) ** 2 for i in range(len(y_pred))]) / 2

# Calcula o erro quadrático médio total.
def mse(X, y, y_pred):
    error = []
    for i in range(len(X)):
        error.append(individual_mse(y[i], y_pred[i]))    
    
    #error = np.array(error)
    return np.mean(error)

# Prediz uma classe para cada elemento de X.
def predict(X, network):
    task = 'r'
    y_pred = []
    for i in range(len(X)):
        predicted = forward_propagate(X[i], network)
        y_pred.append(predicted[len(network)])

    # Se for regressão, o resultado da camada de saída é normalizado para [0,1]
    if task == 'r':
        minmaxscaler = MinMaxScaler()
        y_pred = minmaxscaler.fit_transform(y_pred)

    return y_pred

# Treinamento da rede.
def fit(task, X, y, network, l_rate, epochs, alpha, epsilon, previous_delta):
    if task == 'c':
        scores = list()

    for i in range(epochs):
        for j in range(len(X)):
            new_input = X[j]
            y_pred = forward_propagate(new_input, network)
            network, previous_delta = backward_propagate_error(X[j], y_pred, y[j], network, l_rate, alpha, previous_delta)

        predicted = predict(X, network)
        error = mse(X, y, predicted)
        
        if task == 'c':
            scores.append(accuracy_score(y, predicted))
        
        if not i % 100:
            print("Epoch: %d\tmse: %f" %(i, error))

        if error <= epsilon:
            break

    print("alpha %.1f\teta %.1f\nErro no treino %d: %f" % (alpha, l_rate, i, error))
    if task == 'c':
        print("Train accuracy: %.2f" % np.mean(scores))            

    return network

# Avalia a rede de acordo com a tarefa.
def exec_algorithm(task, X, y, hidden, neurons_hidden, l_rate, epochs, alpha, epsilon):
    start = time.time()

    # Divide 2/3 do dataset para treino e 1/3 para teste.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, shuffle = True)

    # Gera pesos e lista de matriz que armazenará o delta da iteração anterior.
    network, previous_delta = initialize_network(hidden, neurons_hidden, len(y_train[0]), len(X_train[0]))        
    
    # Treina a rede.
    network = fit(task, X_train, y_train, network, l_rate, epochs, alpha, epsilon, previous_delta)
    end = time.time()
    
    # Gera predições e printa a acurácia (classificação) ou o erro (regressão).
    predicted = predict(X_test, network)
    if task == 'c':
        score = accuracy_score(y_test, predicted)   
        print("Test accuracy: %0.2f " % score)
    elif task == 'r':
        error = mse(X_test, y_test, predicted)
        print("Erro no teste: %0.2f" % error)
    else:
        sys.exit("Tarefa inválida.\nPor favor, informe a tarefa corretamente (r, para regressão ou c, para classificação).")

    print("Tempo de execução %.2f segundos." % (end - start))

def main():
    # Leitura do arquivo de entrada e conversão para one-hot.
    task = 'r' # Usar 'r' para regressão ou 'c' para classificação.
    X, y = read_data(task)
    
    # Se a tarefa for classificação, gera o onehot das classes
    # Se a tarefa for regressão, normaliza as coordenadas lat e long para [0,1]
    if task == 'c':
        y = to_onehot(y, 3)
    elif task == 'r':
        minmaxscaler = MinMaxScaler()
        y = minmaxscaler.fit_transform(y)
    else:
        sys.exit("Tarefa inválida.\nPor favor, informe a tarefa corretamente (r, para regressão ou c, para classificação).")
    
    # Normalização das features.
    transformer_X = Normalizer().fit(X)
    normalized_X = transformer_X.transform(X)
    
    # Hiperparâmetros.
    epsilon = 0.01
    hidden = 2
    neurons_hidden = 9
    epochs = 5000
    alpha = 0.9
    l_rate = 0.1
    
    # Avaliação da rede de acordo com a tarefa.
    exec_algorithm(task, normalized_X, y, hidden, neurons_hidden, l_rate, epochs, alpha, epsilon)
    
if __name__ == '__main__':
    main()