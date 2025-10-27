# multi-layer-sigmoid-test
Project made to help my understanding of neural network

Currently capable of creating a model by specifying:
    - Input layer size
    - Number of hidden layers and the number of nodes in each layer
    - Number of Output layer nodes

Hidden layer can be added with model.add_layer() and be randomly initialized with model.generateLayer()
    - Added layers will be inserted behind the output layer and the output layer's weights will be adjusted
      accordingly. ie. Old adjacent layer had 4 nodes but this one has 3? Output layer now has 3 weights instead of 4.

Similarly when creating the model, if more than 1 hidden layer is specified, subsequent hidden layers created by model.__init__() will have their node weights adjusted to match the previous layer's node count.

Many of the functionalities of the neuron class are carried over from previous work and likely won't make it through the next few commits. However, 
    - Activation and fitting for sigmoid neurons are present
    - Save and load functionalities are present
    - Random initialization of weights and bias for a neuron are handled by this class

Weights, bias, and network structure can be saved and loaded back
    model.save() copies all data needed to instantiate a model object to an npz in the directory
    model.load() uses this data to build a new model identical to the one saved to the load file
    model.build() is code needed to both instantiate a random model class and load from a file
