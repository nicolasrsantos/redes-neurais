#
#   Redes Neurais SCC5809 - ICMC-USP 2020
#   Exercicio 2 - MLP
#
#   Alunos:         Nícolas Roque dos Santos
#                   Tales Somensi
#

import numpy as np
from random import seed
from random import random

def init_network(n_inputs):
    n_hidden = int(np.log2(n_inputs))
    n_outputs = n_inputs                    # output layer size is the same size as input layer
    hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]      # the +1 is for bias
    output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]     # the +1 is for bias
    network = list()
    network.append(hidden_layer)
    network.append(output_layer)
    return network

# calculate neuron activation for an input
def activate(weights, inputs):
	activation = weights[-1]
	for i in range(len(weights)-1):
		activation += weights[i] * inputs[i]
	return activation

# transfer neuron activation
def transfer(activation):
	return 1.0 / (1.0 + np.exp(-activation))

def forward_propagate(network, row):
	inputs = row
	for layer in network:
		new_inputs = []
		for neuron in layer:
			activation = activate(neuron['weights'], inputs)
			neuron['output'] = transfer(activation)
			new_inputs.append(neuron['output'])
		inputs = new_inputs
	return inputs

def transfer_derivative(output):
	return output * (1.0 - output)

def backward_propagate_error(network, expected):
	for i in reversed(range(len(network))):
		layer = network[i]
		errors = list()
		if i != len(network)-1:
			for j in range(len(layer)):
				error = 0.0
				for neuron in network[i + 1]:
					error += (neuron['weights'][j] * neuron['delta'])
				errors.append(error)
		else:
			for j in range(len(layer)):
				neuron = layer[j]
				errors.append(expected[j] - neuron['output'])
		for j in range(len(layer)):
			neuron = layer[j]
			neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])

def upd_weights(network, row, l_rate):
	for i in range(len(network)):
		inputs = row[:-1]
		if i != 0:
			inputs = [neuron['output'] for neuron in network[i - 1]]
		for neuron in network[i]:
			for j in range(len(inputs)):
				neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
			neuron['weights'][-1] += l_rate * neuron['delta']

def train_run(network, input_row, expected_row, l_rate, n_epoch):
    print('\tfor %d epochs, with %.1f learning rate\n' % (n_epoch, l_rate))
    for epoch in range(n_epoch):
        output = forward_propagate(network, input_row)
        sum_error = sum([(expected_row[i]-output[i])**2 for i in range(len(expected_row))])
        backward_propagate_error(network, expected_row)
        upd_weights(network, input_row, l_rate)
        print('\tepoch %02d, error = %.3f' % (epoch, sum_error))

def run_xor(l_rate, n_epoch):
    xor = [ [0,0], [0,1], [1,0], [1,1] ]
    xor_out = [ [1,0], [0,1], [0,1], [1,0] ]
    network = init_network(len(xor[0]))
    print('\n\tlearn XOR')
    print('\tfor %d epochs, with %.1f learning rate\n' % (n_epoch, l_rate))
    for epoch in range(n_epoch):
        sum_error = 0
        row_index = 0
        for row in xor:
            output = forward_propagate(network, row)
            expected = xor_out[row_index]
            sum_error += sum([(expected[i]-output[i])**2 for i in range(len(expected))])
            backward_propagate_error(network, expected)
            upd_weights(network, row, l_rate)
            row_index += 1
        print('\tepoch %02d, error = %.3f' % (epoch, sum_error))

def run_identity(idN_size, idN_row, l_rate, num_epochs):
    if idN_row >= idN_size:
        return
    idN = np.identity(idN_size)
    network = init_network(idN_size)
    print('\n\tlearn row %d from identity %dx%d' % (idN_row, idN_size,idN_size))
    train_run(network, idN[idN_row], idN[idN_row], l_rate, num_epochs)

seed(1)

learning_rate = 0.5
num_epochs = 20

identity_size = 8
identity_row = 2

run_identity(identity_size, identity_row, learning_rate, num_epochs)
# run_xor(learning_rate, num_epochs)
