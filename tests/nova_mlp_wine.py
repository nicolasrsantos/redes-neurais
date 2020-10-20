import numpy as np

#
# brief:
#           generate weights for every conection between the layers
#
# parameters:
#           layers:     array that represents the number of neurons in each network layer
#           e.g.: [2, 3, 4] is a network with 2 neurous in the input layer, 3 in the middle layer, 4 in the output layer
#
# return:
#           array of matrixes
#               length (rows):  number_of_layers-1 (each row is for a layer, except the output)
#               each row contains a matrix
#                   matrix rows length:         its_layer_size + 1
#                   matrix columns length:      next_layer_size
#
def initialize_weights( layers=[2,2,2] ):
    weights = list()
    # print('\nlen(layers):', len(layers))
    # print()
    for i in range( len(layers)-1 ):    # for every layer, except the output layer
        W_l = np.random.normal( loc=0, scale=0.1, size=(layers[i]+1, layers[i+1]) )     # centered in zero, scale = desvio padrao
        # print('W_l:', W_l)
        # print()
        weights.append( W_l )
    # print( 'weights:', weights )
    return weights

def activation( V ):
    return 1. / ( 1. + np.exp(-V) )     # sigmoid function (logistic)

#
# parameters:
#           weights:    array of weight layers
#           X:          a single row of input matrix
#
def forward_propagate( weights, X ):
    layer_input = X                 # layer_input receives the single row of input matrix
    Y_list = list()
    Y_list.append( layer_input )    # Y_list is appended with the input row (single row of input matrix)
    # print()
    # print('layer_input:', layer_input)
    # print('Y_list:', Y_list)
    # print('weights[0]:', weights[0])

    for i in range( len(weights) ):             # for each layer (each row contains the weights of a layer)

        layer_input = np.concatenate( [ layer_input, [+1] ] )   # add 1 (the bias) in the end of input matrix row

        V_i = np.dot( layer_input, weights[i] ) # layer_input holds an input matrix row with +1 element in the end
                                                #
                                                # weights[i] holds the weights of a layer
                                                #       len(weights):           num of layers
                                                #       len(weights[i]):        num of neurons in layer[i] + 1
                                                #       len(weights[i][j]):     num of weights in neuron[i][j] (same size as next layer ( len(weights[i+1]) ) )

                                                # numpy.dot multiply the following way
                                                #
                                                #   first
                                                #
                                                #       input[0] multiplies all the weights in neuron[0]
                                                #       input[1] multiplies all the weights in neuron[1]
                                                #       goes on until last neuron (last row)
                                                #
                                                #       like this:
                                                #
                                                #       layer_input[0] multiplies weights[i][0][0], weights[i][0][1], weights[i][0][2] --- weights in layer[i] neuron[0]
                                                #       layer_input[1] multiplies weights[i][1][0], weights[i][1][1], weights[i][1][2] --- weights in layer[i] neuron[1]
                                                #       layer_input[2] multiplies weights[i][2][0], weights[i][2][1], weights[i][2][2] --- weights in layer[i] neuron[2]
                                                #       goes on until last neuron (last row)
                                                #
                                                #   second
                                                #
                                                #       sum all the weights with same index in neurons (same column), like:
                                                #
                                                #       product[i][0][0]+           product[i][0][1]+           product[i][0][2]+
                                                #       product[i][1][0]+           product[i][1][1]+           product[i][1][2]+
                                                #       product[i][2][0]+           product[i][2][1]+           product[i][2][2]+
                                                #       goes on until last neuron (last row)

                                                # V_i becomes an 1-D array with length = num of weights in each neuron (same size as next layer, without the bias)

        # print()
        # print('inputs:  %d\tlayer_input:' % len(layer_input), layer_input)
        # print()
        # print('layers:  %d len(weights)' % len(weights))
        # print('neurons: %d len(weights[i])' % len(weights[i]))
        # print('weights: %d len(weights[i][0])' % len(weights[i][0]))
        # print()
        # print('weights:', weights[i])
        # print('V_i:', V_i)

        layer_output = activation( V_i )    # layer_output receives activated V_i. Each position of V_i is slightly changed with logistic function (activation)
                                            # V_i is an 1-D array with length = num of weights in each neuron (same size as next layer, without the bias)
        # print('layer_output:', layer_output)
        Y_list.append(layer_output)     # Y_list is appended with the activated output of each layer
        layer_input = layer_output      # input of next layer = output of the current layer

    # Y_list contains:
    # [0] input row (single row of input matrix)
    # [1..n] the activated output of each layer
    return Y_list

# função para dar um forward_propagate na base de dados
def predict( W, data ):
    outputs = list()
    for X in data:
        Y = forward_propagate( W, X )
        outputs.append( Y[-1] )
    return outputs

# função para aferir acurácia
def evaluate( W, data, t ):
    Y = predict( W, data )
    hits = np.sum( [ np.argmax(Y[i]) == np.argmax(t[i]) for i in range( len(Y) ) ] )
    acc = hits / len(Y)
    return acc

# computação do erro de uma amostra
def compute_mse( y, t ):
    return 1/2 * np.sum( [ (t[i] - y[i])**2 for i in range(len(y)) ] )

# computação do erro de um conjunto de dados
def compute_total_mse( W, data, labels ):
    y = predict( W, data )
    E = [ compute_mse( y[i], labels[i] ) for i in range(len(data)) ]
    return np.mean( E )

# W = initialize_weights( [2,5,2] )
# print( 'Predictions:', predict( W, data ) )
# print( 'Accuracy', evaluate( W, data, labels ) )
# print( 'MSE:', compute_total_mse( W, data, labels ) )

def sigmoid_derive( Y ):
    return Y * (1 - Y)

#
# parameters:
#           weights:    array of weight layers
#           X:          a single row of input matrix
#           labels:     a single row of labels
#           eta:        float value
#
def train_step(weights, X, labels, eta):

    Y = forward_propagate( weights, X )     # Y receives:
                                            # [0] input row (single row of input matrix)
                                            # [1..n] the activated output of each layer

    # print('\n\n\n\n\n\n\n\n')

    for layer in reversed( range(len(weights)) ):           # for each layer (from last hidden to input layer)
                                                            # e.g.: if there are 3 layers, 'layer' will run as 1 an then 0

        # print('\n------------------------------------------------- current layer: %d ---------------------------------------------------------' % layer)

        Y_l = np.concatenate( [ Y[layer], [+1] ] )          # Y_l will become current layer input with one more element (1) in the end
        # print()
        # print('Y_l:', Y_l)
        # print()

        delta = list()
        # print(weights[layer].shape[1])
        for neuron in range( weights[layer].shape[1] ):     # for each neuron on current layer

            # print('\ncurrent neuron:          ', neuron)

            if layer == len(weights)-1:                     # if layer == hidden before output
                s = labels[neuron] - Y[-1][neuron]          # expected - obtained
            else:
                # print('weights[layer+1][neuron]:\n', weights[layer+1][neuron])
                # print('old_delta:               \n', old_delta)
                s = np.sum( weights[layer+1][neuron] * old_delta )

            delta.append( s * sigmoid_derive( Y[layer+1][neuron] ) )
            # print('delta:                   \n', delta)

            # update weights
            for weight_index in range( weights[layer].shape[0] ):
                weights[layer][weight_index,neuron] += eta * delta[neuron] * Y_l[weight_index]

        old_delta = delta
    return weights

#
# parameters:
#           weights:    array of weight matrices
#           data:       input matrix
#           labels:     output matrix
#
def train( weights, data, labels, eta=0.5, epochs=10000, epsilon=0.1 ):
    error = 100.
    epoch = 0
    # print('len(data):', len(data))
    while error > epsilon:
        crazyIndex = np.random.choice( len(data), len(data), replace=False )   # numpy array with data rows indexes in a random order
        for i in range(len(data)):  # for each row in input matrix
            weights = train_step(weights, data[ crazyIndex[i] ], labels[ crazyIndex[i] ], eta)
        error = compute_total_mse(weights, data, labels)
        if not epoch % 100:
            print( 'Epoch: %d, mse: %f' %(epoch, error) )
        if epoch >= epochs:
            break
        epoch += 1
    print(error)
    return weights

# data = np.array( [ [0, 0], [0, 1], [1, 0], [1, 1] ] )
# labels = np.array( [ [1, 0], [0, 1], [0, 1], [1, 0] ] )
# W = initialize_weights( [2, 20, 2] )
# W = train( W, data, labels, eta=0.5, epochs = 10000, epsilon = 0.001 )
# print( 'Train accuracy:', evaluate( W, data, labels ) )
# print( predict( W, data ) )

# Id8 = np.identity( 8 )
# print('Id8:\n', Id8)
# W = initialize_weights( [8, 3, 8] )
# W = train( W, Id8, Id8, epsilon=0.01 )
# print( [ np.argmax(Y) for Y in predict(W, Id8) ] )
# print( evaluate(W, Id8, Id8) )

# Id15 = np.identity( 15 )
# W = initialize_weights( [15, 4, 15] )
# W = train( W, Id15, Id15, epsilon=0.01 )
# print( [ np.argmax(Y) for Y in predict(W, Id15) ] )
# print( evaluate(W, Id15, Id15) )







features = list()
labels = list()

with open('wine.data', 'r') as file:
    for line in file.readlines():
        row = line.strip('\n').split(',')
        features.append( row[1:] )      # appends elements from second column to the last one
        if(int(row[0]) == 1):
            labels.append( [1,0,0] )
        elif(int(row[0]) == 2):
            labels.append( [0,1,0] )
        elif(int(row[0]) == 3):
            labels.append( [0,0,1] )

features = np.array(features, dtype=np.float32)
labels = np.array(labels, dtype=np.int8)

# print('features.shape:', features.shape)
# print('len(features):', len(features))
# print('len(features[0]):', len(features[0]))
# # print('features:\n', features)
# print('labels.shape:', labels.shape)
# print('len(labels):', len(labels))
# print('len(labels[0]):', len(labels[0]))
# # print('labels:', labels)

W = initialize_weights( [ len(features[0]), 3, 3, len(labels[0]) ] )
W = train( W, features, labels, epsilon=0.01 )
print( evaluate(W, features, labels) )









#
