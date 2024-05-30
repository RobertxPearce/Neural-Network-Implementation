import numpy as np  # Import numpy to help with math.

#---------------------------------------
#     Neuron Implementation
#---------------------------------------

"""
Activation function for the neural network.
Using the sigmoid function to map inputs to
a value between 0 and 1.
"""
def sigmoid(x):
    # Function: f(x) = 1 / (1 + e^(-x))
    return 1 / (1 + np.exp(-x))

"""
Class for neuron that will take an input and use
its weights and bias then send that as ouput.
"""
class Neuron:
    # Default constructor for the neuron class.
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
    
    # Function to pass the inputs forward in the network.
    def feedForward(self, inputs):
        # Multiply the inputs and weights then add the neurons bias.
        total = np.dot(self.weights, inputs) + self.bias
        # Pass the sum through an activation function.
        return sigmoid(total)

#---------------------------------------
#     Neural Network Implementation
#---------------------------------------

"""
Class for the neural network:
- 2 inputs
- 1 hidden layer with 2 neurons (h1, h2)
- Output Layer with 1 neuron
- Each neuron has the name weights ([0, 1]) and bias (0)
"""
class OurNeuralNetwork:
    # Default constructor for the neural network.
    def __init__(self):
        weights = np.array([0, 1])
        bias = 0

        self.h1 = Neuron(weights, bias)
        self.h2 = Neuron(weights, bias)
        self.o1 = Neuron(weights, bias)

    # Function to pass the inputs forward in the network.
    def feedForward(self, x):
        out_h1 = self.h1.feedForward(x)
        out_h2 = self.h2.feedForward(x)

        # Output neuron take outputs from hidden layer as input.
        out_o1 = self.o1.feedForward(np.array([out_h1, out_h2]))

        return out_o1

network = OurNeuralNetwork()
x = np.array([2, 3])
print(network.feedForward(x))