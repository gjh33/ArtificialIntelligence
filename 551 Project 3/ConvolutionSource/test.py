from Layer import Layer
from NeuralNetClassifier import NeuralNetClassifier
from neurons.InnerNeuron import InnerNeuron
from neurons.InputNeuron import InputNeuron
from statistic_functions.Sigmoid import Sigmoid

net = NeuralNetClassifier(5, 1)
#net.input_layer.add_bias_term(1)
layer = Layer()
#layer.add_bias_term(1)
for i in range(0,10):
    layer.add_neuron(InnerNeuron(Sigmoid()))
net.add_hidden_layer(layer)
layer2 = Layer()
layer2.add_bias_term(1)
for i in range(0,5):
    layer2.add_neuron(InnerNeuron(Sigmoid()))
net.add_hidden_layer(layer2)
net.fully_connect_hidden_layer(2)
for i in range(0,10000):
    ss = 1
    net.learn_example([1,1,0,0,0], [0], ss)
    net.learn_example([0,0,0,1,1], [1], ss)
    net.learn_example([1,0,1,0,0], [0], ss)
    net.learn_example([1,1,1,0,0], [0], ss)
    net.learn_example([1,0,0,0,0], [0], ss)
    net.learn_example([0,0,1,0,0], [0], ss)
    net.learn_example([0,1,0,0,0], [0], ss)
    net.learn_example([0,0,0,1,0], [1], ss)
    net.learn_example([0,0,0,0,1], [1], ss)
