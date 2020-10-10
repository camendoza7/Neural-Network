"""
Package to create Multi-Layer Perceptrons (MLPs)
by Christopher Mendoza
"""
import numpy as np

#Add necessary functions using numpy arrays as input
def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_d(x):
    return sigmoid(x)*(1 - sigmoid(x)) 
    
def ReLU(x):
    return x * (x > 0)

def ReLU_d(x):
    return (x > 0).astype(int)

def MSE(y, yhat):
    return np.sum((yhat - y)**2)

def MSE_d(y, yhat):
    return 2*(yhat - y)

def CategoricalCrossEntropy(y, yhat):
    return np.sum(-1 * y * np.log(yhat))

def CategoricalCrossEntropy_d(y, yhat):
    return yhat - y

def bias(array):
    return np.insert(array, 0, 1)

class Layer():
    def __init__(self, shape, activation = 'ReLU'):
        """
        Layer intialization

        Parameters
        ----------
        shape : tuple
            Shape for weight matrix.
        activation : string, optional
            String representing activation function for this layer to use. The default is 'sigmoid'.

        """
        self.n = shape[0]
        self.weights = np.random.randn(*shape)*np.sqrt(2/self.n)
        self.z = None
        self.a = None
        self.e = None
        self.activation = globals()[activation]
        self.activation_d = globals()[activation + '_d']
        self.gradient = np.zeros(shape)
        self.old_gradient = np.zeros(shape)
        
class Net():
    def __init__(self, input_size, loss = 'MSE', alpha = 0.1, mu = 0.0):
        """
        Network initialization

        Parameters
        ----------
        input_size : int
            Size of input i.e. number of features to use.
        loss : string, optional
            Loss function to use. The default is 'MSE'.
        alpha : float, optional
            Learning Rate. The default is 0.1.
        mu : float, optional
            Momentum parameter it's off by default. The default is 0.0.

        """
        self.input_size = input_size
        self.loss = globals()[loss]
        self.loss_d = globals()[loss + '_d']
        self.alpha = alpha
        self.mu = mu
        self.layers = []
        self.loss_value = 0
        self.output = None
        
    def add_layer(self, n, activation = 'ReLU'):
        """
        Adds layer to network

        Parameters
        ----------
        n : int
            Number of nodes in the layer.
        activation : string, optional
            String representing the activation function to use in this layer. The default is 'sigmoid'.

        Returns
        -------
        Layer
            Returns newly created layer.

        """
        if not self.layers:
            shape = (n, self.input_size + 1)
        else:
            shape = (n, self.layers[-1].n + 1)
        self.layers.append(Layer(shape, activation = activation))
        return self.layers[-1]
        
    def feed_forward(self, x):
        """
        Does the feed forward process to produce an output.

        Parameters
        ----------
        x : Numpy array
            Input values (excluding bias term).

        Returns
        -------
        Numpy array
            The output of the network.

        """
        #Compute z and a for each layer
        for i, l in enumerate(self.layers):
            #First layers uses input plus a bias term
            if i == 0:
                l.z = l.weights @ bias(x)
            #Other layers use a of previous layer plus a bias term
            else:
                l.z = l.weights @ bias(self.layers[i - 1].a)
            #Compute the output of the activation function    
            l.a = l.activation(l.z)
        self.output = self.layers[-1].a
        return self.output
            
    def backprop(self, x, y):
        """
        Performs backpropogation given one sample.

        Parameters
        ----------
        x : Numpy array
            Sample input.
        y : Numpy array or float
            Target value for x.

        Returns
        -------
        list
            List containing the gradients for each layer.

        """
        #Go through the layers in reverse order, the index from enumeration is also reversed for simplicity
        for i, l in reversed(list(enumerate(self.layers))):
            #Last layer in network
            if i == len(self.layers) - 1:
                #Compute the loss and gradient for the last layer, the e term is saved for computation of dc/da of hidden layers
                self.loss_value = self.loss(y, l.a)
                l.e = l.activation_d(l.z) * self.loss_d(y, l.a)
                l.gradient += bias(self.layers[i - 1].a) * np.tile(l.e, (self.layers[i - 1].n + 1, 1)).T
            #Hidden layers
            else:
                #Extend the e term in the next layer to make computation simpler
                ee = np.tile(self.layers[i + 1].e, (l.n,1)).T
                #Multiply the weights to compute dc/da
                dcda = self.layers[i + 1].weights[:, 1:] * ee 
                #The summing term in dcda for hidden layers
                l.e = dcda.sum(axis = 0)
                #Extend dc/da to the appropriate shape to make computation of the gradient simpler
                if i == 0:
                    dcda = np.tile(l.e, (self.input_size + 1,1)).T
                    l.gradient += sigmoid_d(l.z).reshape(l.n,1) @ bias(x).T.reshape(1, self.input_size + 1) * dcda
                else:
                    dcda = np.tile(l.e, (self.layers[i - 1].n + 1,1)).T
                    l.gradient += sigmoid_d(l.z).reshape(l.n,1) @ bias(self.layers[i - 1].a).T.reshape(1, self.layers[i - 1].n + 1) * dcda     
        return [k.gradient for k in self.layers]
                
    def update_weights(self):
        """
        Updates weights.

        Returns
        -------
        list
            List containing the new weights for each layer

        """
        for l in self.layers:
            v = self.mu * l.old_gradient + self.alpha * l.gradient
            l.weights -= v
            l.old_gradient = v
            l.gradient *= 0
        return [k.weights for k in self.layers]          
    
    def fit(self, X, y, minibatch = 1, epochs = 50):
        """
        Fits the network based on training data provided. A minibatch can be specified by default it is SGD.

        Parameters
        ----------
        X : numpy array
            Training data.
        y : numpy array
            Target data for each sample.
        minibatch : int, optional
            Size of minibatch. The default is 1.
        epochs : int, optional
            Number of epochs to train for. The default is 50.

        Returns
        -------
        None.

        """
        for epoch in range(1, epochs + 1):
            print(f'Currently on epoch {epoch}')
            bc = 0
            for xs, ys in zip(X,y):
                self.feed_forward(xs)
                self.backprop(xs, ys)
                bc += 1
                if bc == minibatch:
                    for l in self.layers:
                        l.gradient /= minibatch
                    self.update_weights()
                    bc = 0
                    
    def predict(self, X):
        """
        Returns the output of the network given input data.

        Parameters
        ----------
        X : numpy array
            Samples to predict on.

        Returns
        -------
        numpy array
            Output of the network for each sample.

        """
        outputs = []
        for x in X:
            outputs.append(self.feed_forward(x))
        return np.vstack(outputs)
        