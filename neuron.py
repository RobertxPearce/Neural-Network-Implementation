import numpy as np  # Import numpy to help with math.

#-------------------------------
#     Neuron Implementation
#-------------------------------

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

weight = np.array([0,1])    # w1 = 0 | w2 = 1
bias = 4                    # b = 4
n = Neuron(weight, bias)    # Neuron object.

x = np.array([2, 3])        # x1 = 2 | x2 = 3
print(n.feedForward(x))    # 0.9990889488055994
