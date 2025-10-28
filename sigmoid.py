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
            # Defunct
            self.load(model)
        elif np.all(self.weights == 0) and self.bias == 0.0:
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
# node_params = a list of tuples containing neuron information: weights[], float(bias)
class layer:
    def __init__(self, node_density: int, input_size : int = 2, node_params: list[tuple] = None):
        # NOTE: Could take a node list for the layer and pass it to neuron for initialization
        # If the node list section is None, randomize nodes instead
        self.nodes = []
        self.size = node_density
        self.weights_per_node = input_size


        # Initialize the node list with randomly generated neurons
        for i in range(node_density):
            if node_params is not None:
                # Initialize from provided tuple list
                n = neuron(weights = node_params[i][0], bias = float(node_params[i][1]))
            else:
                # Initialize with no params (randomized by neuron init)
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

    # NOTE: (FIXED) Loading a model from npz does not correctly set the model parameters.
    # Build is being called to set up the model architecture which is overwirting node values.

    # Will initialize a model with
    # model = an npz file with weights and biases to load
    def __init__(self, model_npz=None, input_size=2, output_size=1, hidden_layers=1, nodes_in_hidden=2):
        if model_npz is not None:
            self.load(model_npz)
            print("Model loaded from NPZ.")
        else:
            print("Model was None.")       
            self.layers = []
            self.input_size = input_size
            self.hidden_layers = hidden_layers
            # FIXME: Look for better solution to autosizing / layer width management
            self.nodes_per_hidden = nodes_in_hidden if nodes_in_hidden > 0 else 1
            self.output_size = output_size
            self.build()

    # Builds the model architecture from instantiated parameters
    # Fully internal but can be called to replicate the model and append it to itself
    # node_params = A list of tuples each containing weights[] and float(bias) for a neuron in a layer
    # layer_sizes = A list of ints that specify nodes per layer if used correctly FIXME
    def build(self, node_params: list[tuple] = None, layer_sizes: list[int] = None):
        if layer_sizes is None:
            layer_sizes = [self.nodes_per_hidden for i in range(self.hidden_layers)] + [self.output_size]
        # Create hidden layers
        # Layer 1
        # cumulative_index keeps track of where we should start the range in node_params
        cumulative_index = 0
        hidden_layer = layer(layer_sizes[0], self.input_size, node_params[0: layer_sizes[0]] if node_params is not None else None)
        self.layers.append(hidden_layer)
        cumulative_index += self.layers[0].size
        # Next layers must meet certain conditions if present.
        # as many weights as there are nodes in previous layer
        for i in range(self.hidden_layers - 1):
            hidden_layer = layer(layer_sizes[i + 1], self.layers[i].size, node_params[cumulative_index: cumulative_index + layer_sizes[i+1]] if node_params is not None else None)
            self.layers.append(hidden_layer)
            cumulative_index += self.layers[-1].size

        # Create output layer
        # NOTE: same rule for weights and previous nodes still applies
        # FIXME: Output does not allow for multiple nodes
        output_layer = layer(self.output_size, self.layers[-1].size, node_params[cumulative_index: cumulative_index + layer_sizes[-1]] if node_params is not None else None)
        self.layers.append(output_layer)

    # Generates and returns a layer with randomly generated nodes
    # node_density = the number of nodes desired for layer
    # input_size = the number of weights desired for wach node
    def generate_layer(self, node_density: int, input_size):
        new_layer = layer(node_density, input_size)
        return new_layer

    # Add a layer to model layers before the output layer
    # l - must be an existing layer object
    # TODO: merge with generate_layer
    def add_layer(self, l: layer):
            # Insert before output layer
            self.layers.insert((len(self.layers) - 1), l)
            self.hidden_layers += 1

            # Adjust output layer's weights to match the node count of the new adjacent layer
            self.layers.pop()
            # New layer from scratch
            output_layer = layer(self.output_size, self.layers[-1].size)
            self.layers.append(output_layer)

    # Accumulates necessary data to instantiate and saves it to an npz
    # FIXME: The addition of layers casuing errors in loading is starting with save
    def save(self, model_name_npz):
        data = {}
        # Base model parameters
        data[f"input_size"] = self.input_size
        data[f"output_size"] = self.output_size
        data[f"hidden_layers"] = self.hidden_layers
        data[f"nodes_per_hidden"] = self.nodes_per_hidden

        #print(self.output_size)

        # Layers and node information
        for i, layer in enumerate(self.layers):
            # Save each layer's size
            data[f"L{i}_size"] = layer.size
            for j, node in enumerate(layer.nodes):
                data[f"L{i}_N{j}_weights"] = node.weights
                data[f"L{i}_N{j}_bias"] = node.bias
        np.savez(model_name_npz, **data)
        print(f"Model saved as {model_name_npz}!")

    # Loads a model from an npz
    # NOTE: This is currently just a copy of half of the constructor      
    def load(self, model_name_npz):
        # Cycle through model data
        data = np.load(model_name_npz)
        self.input_size = int(data["input_size"])
        self.output_size = int(data["output_size"])
        self.hidden_layers = int(data["hidden_layers"])
        self.nodes_per_hidden = int(data["nodes_per_hidden"])
        self.layers = []

        node_params = []
        layer_sizes = []
        # Hidden layers
        for i in range(self.hidden_layers):
            layer_sizes.append(int(data[f"L{i}_size"]))
            for j in range(int(data[f"L{i}_size"])):  #   FIXME Here is the problem with add_layer FIXME   #
                # Load each node's weights and bias
                weights = data[f"L{i}_N{j}_weights"]
                bias = data[f"L{i}_N{j}_bias"]
                node_params.append((weights, bias))
        
        # Output layer
        layer_sizes.append(int(data[f"L{self.hidden_layers}_size"]))
        for j in range(self.output_size):
            weights = data[f"L{self.hidden_layers}_N{j}_weights"]
            bias = data[f"L{self.hidden_layers}_N{j}_bias"]
            node_params.append((weights, bias))

        self.build(node_params, layer_sizes)

        print(f"Model {model_name_npz} loaded!")

    def fit(self, features, labels):
        # Train the model using the provided features and labels
        for i, feature in enumerate(features):
            self.forward(feature)

    def forward(self, features):
        # Perform a forward pass through the model using the provided features
        first_pass = []
        for l in self.layers:
            for n in layer.nodes:
                first_pass.append(n.activate(features))
        
        next_pass = []
        for i, l in enumerate(self.layers):
            if i == 0: continue 
            for j, n in enumerate(l.nodes):
                next_pass.append(n.activate(first_pass))
            first_pass = next_pass
            next_pass = []

    def backpropagate(self, labels):
        # Perform backpropagation to update model weights based on the provided labels
        pass

    # Run a single set of features through the entire model without training
    def dry_run(self, features):
        # Perform a forward pass without training
        pass


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
    # NOTE: Instantiating from scratch
    # FIXME: (FIXED - Issue was wide misuse of nodes_per_hidden) adding a layer to class constructor is doing something wrong in save/load?
    # TODO: Autosizing on generate_layer / add_layer
    m = model(None, 2, 1, 1, 2)
    m.add_layer(m.generate_layer(2, m.layers[len(m.layers)-2].size))
    m.describe_verbose()

    # NOTE: Svaing and loading a model
    #m.save("test.npz")
    #m.describe_verbose()
    #m.load("test.npz")
    #m.describe_verbose()

    # NOTE: Instantiating straight from npz
    #m2 = model("test.npz")
    #m2.describe_verbose()

if __name__ == "__main__":
    main()