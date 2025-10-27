import numpy as np
import pandas as pd

# Defines a single node in a layer of a model
class neuron:
    def __init__(self, weights, bias, model = "none"):
        # Weights should be provided as a list and bias as a float
        # Weighted sum is only calculated in self.summate() and acts as the activation input
        # n is the learning rate for the neuron
        # error will be calculated as the difference between a label and this neuron's prediction
        self.weights = weights
        self.bias = bias
        self.weighted_sum = 0.0
        self.pred = 0.0
        self.n = 0.1
        self.error = 0.0

        if model != "none":
            # Attempt to load existing neuron weights and bias from npz
            self.load(model)
        else:
            # Randomize weights and bias
            self.initialize()

    # Sets all weights to a random value
    def initialize(self):
        self.weights = np.random.uniform(-1,1,size=len(self.weights))
        self.bias = np.random.uniform(-1,1)

    # Calculates the dot product of weights and inputs then adds it's bias
    # x - the inputs which should be equal to the number of weights
    # The weighted sum will become the input for sigmoid activation
    def summate(self, x):
        self.weighted_sum = np.dot(self.weights, x) + self.bias
        return self.weighted_sum

    # Sigmoid activation function
    # x - the inputs which should be equal to the number of weights
    # summate is called automatically within this function
    def activate(self, x):
        return (1 / (1 + np.exp(-self.summate(x))))

    # Training loop for learning dataset
    # features = a single list equal in size to weights
    # label = the expected output for the features provided
    def fit(self, features, label):
        self.pred = self.activate(features)
        self.error = label - self.pred

        # Differentiate with respect to activation?
        # NOTE: Will soon be made defunct with generalized backpropagation
        predPrime = self.pred * (1-self.pred)

        # Correct values
        self.weights += self.n * self.error * predPrime * features
        self.bias += self.n * self.error * predPrime
        return self.pred

    def predict(self, features):
        self.pred = self.activate(features)
        return self.pred

    # Saves neuron to npz
    def save(self, npz):
        np.savez(npz, weights = self.weights, bias = self.bias)
        print(f"{npz} saved!")

    # Loads weights and bias from npz file assuming some formatting
    def load(self, model_npz):
        model_data = np.load(model_npz)
        self.weights = model_data["weights"]
        self.bias = model_data["bias"]

    def describe(self):
        weights = ""
        for i, weight in enumerate(self.weights):
            weights += (f"    W{i}={weight}\n")
        bias = f"    Bias = {self.bias}\n\n"
        return weights + bias

# Handles perceptrons found in a layer of a model
# nodes[] = A list containing all neuron objects present in the layer
# size = the number of neurons contained in nodes[]
class layer:
    def __init__(self, node_density: int, input_size : int = 2):
        self.nodes = []
        self.size = node_density
        self.weights_per_node = input_size

        # Initialize the node list with randomly generated neurons
        for i in range(node_density):
            n = neuron(weights = np.zeros(input_size), bias = 0.0)
            self.nodes.append(n)
    
    def describe(self, index = -1):
        # Describe all nodes in layer
        if index == -1:
            descriptions = ""
            for i, node in enumerate(self.nodes):
                descriptions += (f" Node {i}:\n")
                descriptions += node.describe()
            return descriptions
        # Describe single node in layer
        else:
            if index < self.size:
                return self.nodes[index].describe()
            else:
                return "Index out of range"

# input_size = the expected size of the input layer
#   determines the number of weights that a node will hold
# output_size = the desired number of nodes in the output layer
# hidden_layers = the desired number hidden layers
class model:
    # A call will run like: 
    #   - How many inputs are there, how many outputs do I want?
    #   - How many hidden layers do I want, how many nodes in the hidden layers?

    # NOTE: (FIXED) New issue, if there is more than one hidden layer, the input size of the second hidden layer
    # must be equal to the number of nodes in the first hidden layer.
    # This applies to all subsequent hidden layers as well.

    # NOTE: (FIXED) New issue, adding a layer after initialization will not adjust the weights_per_node of the output layer.

    # Will initialize a model with a number of hidden layers and a 1 node output layer based on params
    def __init__(self, input_size, output_size, hidden_layers = 1, nodes_in_hidden = 2):
        self.layers = []
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.nodes_per_hidden = nodes_in_hidden if nodes_in_hidden > 0 else 1
        self.output_size = output_size

        # Create hidden layers
        # Layer 1
        hidden_layer = layer(self.nodes_per_hidden, self.input_size)
        self.layers.append(hidden_layer)
        # Next layers must meet certain parameters if present.
        # ie as many weights as there are nodes in previous layer
        for i in range(self.hidden_layers - 1):
            hidden_layer = layer(self.nodes_per_hidden, self.layers[i].size)
            self.layers.append(hidden_layer)

        # Create output layer
        # NOTE: a layers nodes must have as many weights as there are nodes in the previous layer
        output_layer = layer(1, self.nodes_per_hidden)
        self.layers.append(output_layer)

    # Generates and returns a layer with randomly generated nodes
    # node_density = the number of nodes desired for layer
    # input_size = the number of weights desired for wach node
    def generate_layer(self, node_density: int, input_size):
        new_layer = layer(node_density, input_size)
        return new_layer

    # Add a layer to model layers before the output layer
    # l - must be an existing layer object
    def add_layer(self, l: layer):
            # Insert before output layer
            self.layers.insert((len(self.layers) - 1), l)
            self.hidden_layers += 1

            # Adjust output layer's weights to match the node count of the new adjacent layer
            self.layers.pop()
            output_layer = layer(1, self.layers[-1].size)
            self.layers.append(output_layer)

    def describe_verbose(self):
        print(f"Number of layers: {len(self.layers)}")
        print(f" Hidden: {self.hidden_layers} layer with {self.nodes_per_hidden} nodes each")
        print(f" Output: {self.output_size} node")
        print("\nLayers:")
        for i, layer in enumerate(self.layers):
            if i == len(self.layers) - 1:
                print("Output Layer:")
            else:
                print(f"Hidden Layer {i}:")
            print(layer.describe())

def main():
    # two inputs, one output, one hidden layer with two nodes
    m = model(2, 1, 1, 2)
    #m.describe()

    m.add_layer(m.generate_layer(2, m.layers[len(m.layers)-2].size))
    m.describe_verbose()


if __name__ == "__main__":
    main()