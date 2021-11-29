
import matplotlib.pyplot as plt
import numpy as np
import math
import time
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

import Loading_Datasets

import random

class Network:
    def __init__(self, input_layer, hidden_layer, output_layer, activation_function):
        self.input_layer = input_layer  # number of neurons in the input layer
        self.hidden_layer = hidden_layer  # number of neurons in the hidden layers (represented by an array of integers)
        self.output_layer = output_layer  # number of neurons in the output layer
        self.activation_function = activation_function
        self.weights = [np.zeros((self.hidden_layer[0], self.input_layer)),np.zeros((self.hidden_layer[1], self.hidden_layer[0])),np.zeros((self.output_layer, self.hidden_layer[1]))]
        self.biases = [np.zeros((self.hidden_layer[0],1)),np.zeros((self.hidden_layer[1],1)),np.zeros((self.output_layer,1))]
        self.neuron_values = []
        self.z = []
        self.grad_w = []
        self.grad_b = []

    def fill_weights_and_biases(self):
        if len(self.hidden_layer) == 0:  # if there is no hidden layer
            self.fill_weights_and_biases_without_hidden_layer()
        else:
            self.fill_weights_and_biases_with_hidden_layer()

    def fill_weights_and_biases_without_hidden_layer(self):
        n_i = self.input_layer
        n_o = self.output_layer
        # creates a matrix of weights, corresponding to two adjacent layers
        weight_matrix = np.random.normal(0, 1, (n_o, n_i))
        self.weights.append(weight_matrix)
        # creates a vector of biases, corresponding to two adjacent layers
        bias_vector = np.random.normal(0, 1, (n_o, 1))
        self.biases.append(bias_vector)

    def fill_weights_and_biases_with_hidden_layer(self):
        n_i = self.input_layer
        n_o = self.output_layer
        h = self.hidden_layer
        for i in range(len(self.hidden_layer) + 1):
            if i == 0:
                # creates a matrix of weights, corresponding to two adjacent layers
                weight_matrix = np.random.normal(0, 1, (h[i], n_i))
                self.weights[i] = weight_matrix
                # self.weights.append(weight_matrix)
                # creates a vector of biases, corresponding to two adjacent layers
                bias_vector = np.random.normal(0, 1, (h[i], 1))
                self.biases[i] = bias_vector
            elif i != len(self.hidden_layer):
                # creates a matrix of weights, corresponding to two adjacent layers
                weight_matrix = np.random.normal(0, 1, (h[i], h[i - 1]))
                self.weights[i] = weight_matrix
                # self.weights.append(weight_matrix)
                # creates a vector of biases, corresponding to two adjacent layers
                bias_vector = np.random.normal(0, 1, (h[i], 1))
                # self.biases.append(bias_vector)
                self.biases[i] = bias_vector

            else:
                # creates a matrix of weights, corresponding to two adjacent layers
                weight_matrix = np.random.normal(0, 1, (n_o, h[i - 1]))
                self.weights[i] = weight_matrix
                # self.weights.append(weight_matrix)
                # creates a vector of biases, corresponding to two adjacent layers
                bias_vector = np.random.normal(0, 1, (n_o, 1))
                self.biases[i] = bias_vector
                # self.biases.append(bias_vector)


    def sigmoid(self, x):
        return 1 / (1 + math.e ** -x)

    def sigmoid_prime(self, x):
        sig = self.sigmoid(x)
        return sig * (1 - sig)

    # determine the values of the next layer, based on the values of the current layer
    def feed_forward(self, a, i):
        if i == 0:
            # print("+++++++")
            self.neuron_values.append(a)  # appending the values of the initialized input neurons
            self.z.append(a)
        if self.activation_function == "sigmoid":
            w = self.weights[i]
            b = self.biases[i]
            tmp = np.dot(w, a) + b
            a = self.sigmoid(tmp)
        self.z.append(tmp)
        self.neuron_values.append(a)
        return a

    def forward_propagation(self, a):
        self.neuron_values = []  # reset
        self.z = []  # reset
        if len(self.hidden_layer) == 0:  # if there is no hidden layer
            return self.forward_propagation_without_hidden_layer(a)
        else:
            return self.forward_propagation_with_hidden_layer(a)

    def forward_propagation_without_hidden_layer(self, a):
        return self.feed_forward(a, 0)

    def forward_propagation_with_hidden_layer(self, a):
        for i in range(len(self.hidden_layer) + 1):
            a = self.feed_forward(a, i)

        
        return a
#######################################################################

    def feed_backward(self, a, data, i, a_gradients=None):
        w_gradients = np.zeros(self.weights[i - 1].shape)
        b_gradients = np.zeros(self.biases[i - 1].shape)
        new_a_gradients = np.zeros((len(a[i - 1]), 1))  # for the next round
        for j in range(len(a[i])):
            if a_gradients is not None:
                b_gradient = a_gradients[j] * self.sigmoid_prime(a[i][j])
                b_gradients[j] = b_gradient
            else:
                b_error = 2 * (a[i][j] - data[2 + j])  # only for the last layer
                b_gradient = b_error * self.sigmoid_prime(a[i][j])
                b_gradients[j] = b_gradient
            for m in range(len(self.weights[i - 1][j])):
                if a_gradients is not None:
                    w_gradient = a_gradients[j] * self.sigmoid_prime(a[i][j]) * a[i - 1][m]
                    a_gradient = a_gradients[j] * self.sigmoid_prime(a[i][j]) * self.weights[i - 1][j][m]
                else:
                    w_error = 2 * (a[i][j] - data[2 + j])  # only for the last layer
                    w_gradient = w_error * self.sigmoid_prime(a[i][j]) * a[i - 1][m]
                    a_gradient = w_error * self.sigmoid_prime(a[i][j]) * self.weights[i - 1][j][m]
                new_a_gradients[m][0] += a_gradient
                w_gradients[j][m] = w_gradient
        if len(self.hidden_layer) == 0:
            gradients = w_gradients, b_gradients
        else:
            gradients = w_gradients, b_gradients, new_a_gradients
        return gradients


    def backward_propagation(self, data):
        a = self.neuron_values
        if len(self.hidden_layer) == 0:  # if there is no hidden layer
            return self.backward_propagation_without_hidden_layer(a, data)
        else:
            return self.backward_propagation_with_hidden_layer(a, data)

    def backward_propagation_without_hidden_layer(self, a, data):
        return self.feed_backward(a, data, 1)

    def backward_propagation_with_hidden_layer(self, a, data):
        layer_gradients = []
        a_gradients = None
        for i in range(len(self.hidden_layer) + 1, 0, -1):
            w_gradients, b_gradients, a_gradients = self.feed_backward(a, data, i, a_gradients)
            layer_gradients.append((w_gradients, b_gradients))
        return layer_gradients

    def backward_propagation_gradiant(self,data,result,label):

        gradient_w1 = np.zeros((self.hidden_layer[0], self.input_layer))
        gradient_w2 = np.zeros((self.hidden_layer[1], self.hidden_layer[0]))
        gradient_w3 = np.zeros((self.output_layer, self.hidden_layer[1]))
        gradient_a2 = np.zeros((self.hidden_layer[1],1))
        gradient_a1 = np.zeros((self.hidden_layer[0],1))
        
        gradient_b1 = np.zeros((self.hidden_layer[0],1))
        gradient_b2 = np.zeros((self.hidden_layer[1],1))
        gradient_b3 = np.zeros((self.output_layer,1))

        for layer in range(len(self.hidden_layer)+1,0,-1):
            if layer == 1:
                sigprim_z1 = []
                for x in self.z[1]:
                    sigprim_z1.append(self.sigmoid_prime(x[0]))
                for m in range(self.hidden_layer[0]):
                        gradient_b1[m,0] += gradient_a1[m,0] * sigprim_z1[m]

                for m in range(self.hidden_layer[0]):
                    for v in range(self.input_layer):
                        gradient_w1[m][v] += gradient_b1[m,0] * self.neuron_values[0][v]


            if layer == 2:
                sigprim_z2 = []
                for x in self.z[2]:
                    # print(x[0])
                    sigprim_z2.append(self.sigmoid_prime(x[0]))
                for k in range(self.hidden_layer[1]):
                    gradient_b2[k,0] += gradient_a2[k,0] * sigprim_z2[k]
                
                for k in range(self.hidden_layer[1]):
                    for m in range(self.hidden_layer[0]):
                        gradient_w2[k, m] += gradient_b2[k,0] * self.neuron_values[1][m]

                for k in range(self.hidden_layer[0]):
                    for j in range(self.hidden_layer[1]):
                        gradient_a1[k,0] += gradient_b2[j,0] * self.weights[layer-1][j][k]


            if layer == 3:
                sigprim_z3 = []
                for x in self.z[3]:
                    sigprim_z3.append(self.sigmoid_prime(x[0]))
                for j in range(self.output_layer):
                    gradient_b3[j,0] += 2 * sigprim_z3[j] * (result[j] - label[j])
                for j in range(self.output_layer):
                    for k in range(self.hidden_layer[1]):
                        gradient_w3[j, k] += gradient_b3[j,0] * self.neuron_values[2][k]
                for k in range(self.hidden_layer[1]):
                    for j in range(self.output_layer):
                        gradient_a2[k,0] += gradient_b3[j,0] * self.weights[layer-1][j][k]

        self.grad_w = [gradient_w1,gradient_w2,gradient_w3]
        self.grad_b = [gradient_b1,gradient_b2,gradient_b3]
        return self.grad_w,self.grad_b







    def backward_propagation_gradiant_Vectorization(self,data,result,label):

        gradient_w1 = np.zeros((self.hidden_layer[0], self.input_layer))
        gradient_w2 = np.zeros((self.hidden_layer[1], self.hidden_layer[0]))
        gradient_w3 = np.zeros((self.output_layer, self.hidden_layer[1]))
        gradient_a2 = np.zeros((self.hidden_layer[1],1))
        gradient_a1 = np.zeros((self.hidden_layer[0],1))
        
        gradient_b1 = np.zeros((self.hidden_layer[0],1))
        gradient_b2 = np.zeros((self.hidden_layer[1],1))
        gradient_b3 = np.zeros((self.output_layer,1))

        for layer in range(len(self.hidden_layer)+1,0,-1):
            if layer == 1:
                sigprim_z1 = []
                for x in self.z[1]:
                    sigprim_z1.append(self.sigmoid_prime(x[0]))
                for m in range(self.hidden_layer[0]):
                        gradient_b1[m,0] += gradient_a1[m,0] * sigprim_z1[m]

                for m in range(self.hidden_layer[0]):
                    for v in range(self.input_layer):
                        gradient_w1[m][v] += gradient_b1[m,0] * self.neuron_values[0][v]


            if layer == 2:
                sigprim_z2 = []
                for x in self.z[2]:
                    # print(x[0])
                    sigprim_z2.append(self.sigmoid_prime(x[0]))
                for k in range(self.hidden_layer[1]):
                    gradient_b2[k,0] += gradient_a2[k,0] * sigprim_z2[k]
                
                for k in range(self.hidden_layer[1]):
                    for m in range(self.hidden_layer[0]):
                        gradient_w2[k, m] += gradient_b2[k,0] * self.neuron_values[1][m]

                for k in range(self.hidden_layer[0]):
                    for j in range(self.hidden_layer[1]):
                        gradient_a1[k,0] += gradient_b2[j,0] * self.weights[layer-1][j][k]


            if layer == 3:
                sigprim_z3 = []
                for x in self.z[3]:
                    sigprim_z3.append(self.sigmoid_prime(x[0]))
                for j in range(self.output_layer):
                    gradient_b3[j,0] += 2 * sigprim_z3[j] * (result[j] - label[j])
                for j in range(self.output_layer):
                    for k in range(self.hidden_layer[1]):
                        gradient_w3[j, k] += gradient_b3[j,0] * self.neuron_values[2][k]
                for k in range(self.hidden_layer[1]):
                    for j in range(self.output_layer):
                        gradient_a2[k,0] += gradient_b3[j,0] * self.weights[layer-1][j][k]

        self.grad_w = [gradient_w1,gradient_w2,gradient_w3]
        self.grad_b = [gradient_b1,gradient_b2,gradient_b3]
        return self.grad_w,self.grad_b








#######################################################################
    def plot_single_point(self, point, a):
        # # multi neuron output
        # max = 0
        # best_i = 0
        # for i in range(len(a)):
        #     if a[i] >= max:
        #         max = a[i]
        #         best_i = i
        # plt.scatter(data[0], data[1], color=colors[best_i])

        # single neuron output
        if a[0] < 0.5:
            color = 'r'
        else:
            color = 'b'
        plt.scatter(point[0], point[1], color=color)

    def show_result(self, points, results):
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='c1 (red)', markerfacecolor='r', markersize=10),
            Line2D([0], [0], marker='o', color='w', label='c2 (blue)', markerfacecolor='b', markersize=10),
        ]
        fig, ax = plt.subplots()

        for point, result in zip(points, results):
            self.plot_single_point(point, result)

        plt.xlabel('x1')
        plt.ylabel('x2')
        ax.legend(handles=legend_elements, loc='lower right')
        plt.show()

    def print_accuracy(self, test_data, results):
        score = 0
        i = 0
        for data in test_data:
            flattened  = [val for sublist in data[1] for val in sublist]
            if results[i] == flattened.index(max(flattened)):
                score += 1
            i += 1
        return score








def main(part):
    print('Welcome to the Artificial Neural Network Classifier!')
    train_data , test_data = Loading_Datasets.loading_dataset()
    network = Network(102, [150,60], 4, "sigmoid")
    network.fill_weights_and_biases()

    if part == 1:
        results = []
        for data in train_data[:200]:
            result = network.forward_propagation(data[0])
            flattened  = [val for sublist in result for val in sublist]
            results.append(flattened.index(max(flattened)))

        print("accuracy is {}%".format(network.print_accuracy(train_data[:200], results)/float(2)))


    if part == 2:
        epoch=10
        batch_size=10
        learning_rate=1
        costs = []
        ts = train_data[:200]
        error = []

        for i in range(epoch):
            epoch_cost = 0
            epoch_correct_res = 0
            random.shuffle(ts)  
            # ts = train_data
            batched_train = [ts[j:j + batch_size] for j in range(0, len(ts), batch_size)]            
            print("*********************************************")
            sum = 0
            for batch in batched_train:
                w_gradients = []
                b_gradients = []
                for k in range(len(network.hidden_layer)+1):
                    w_gradients.append(np.zeros_like(network.weights[k]))
                    b_gradients.append(np.zeros_like(network.biases[k]))
 

                
                results = [] #contain indices of maximum element from forward_propagation
                # return
                for data in batch:
                    flattened_labels  = [val for sublist in data[1] for val in sublist]
                    label = flattened_labels.index(max(flattened_labels))
                    result = network.forward_propagation(data[0])

                    # print("***")

                    flattened_result  = [val for sublist in result for val in sublist]

                    results.append(flattened_result.index(max(flattened_result)))

                    s = list(np.array(flattened_result) - np.array(flattened_labels))
                    # print(s)

                    sum += (np.dot(s,s))

                    # result as list of one element in list
                    gw,gb = network.backward_propagation_gradiant(data,flattened_result,flattened_labels)

                    # return
                    for k in range(len(network.hidden_layer)+1):
                        w_gradients[k] += gw[k]
                        b_gradients[k] += gb[k]


                for k in range(0, len(network.hidden_layer)+1):
                    network.weights[k] = network.weights[k] - (learning_rate * (w_gradients[k]/batch_size))
                    network.biases[k] = network.biases[k] - (learning_rate * (b_gradients[k]/len(batch)))
            error.append(sum/float(200))
                # print("accuracy is {}%".format(network.print_accuracy(batch, results)*100/float(len(batch))))


            results = []
            for data in ts:
                result = network.forward_propagation(data[0])
                flattened  = [val for sublist in result for val in sublist]
                results.append(flattened.index(max(flattened)))

            print("*******accuracy is {}%".format(network.print_accuracy(ts, results)/float(2)))
    
        epoch_size = [x for x in range(epoch)]
        plt.plot(epoch_size, error)
        plt.savefig("1.png")
if __name__ == "__main__":
    # print("hello")
    main(2)