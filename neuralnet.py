"""
Package to create Multi-Layer Perceptrons (MLPs)
by Christopher Mendoza
"""
import numpy as np
from sklearn.model_selection import train_test_split

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
    return np.mean((yhat - y)**2)

def MSE_d(y, yhat):
    return 2*(yhat - y)

def CategoricalCrossEntropy(y, yhat):
    return np.sum(-1 * y * np.log(yhat))

def CategoricalCrossEntropy_d(y, yhat):
    return yhat - y

def bias_transform(X):
    #Flatten to make sure reshape is in right order
    r = X.shape[0]
    c = X.shape[1] + 1
    temp = np.hstack([np.ones((r, 1)), X]).flatten()
    return temp.reshape(r, c, 1)

def bias(a):
    return np.insert(a, 0, 1, axis = 1)

#Function to make mini-batches
def make_batches(X, y, batch_size):
    for b in range(0, len(X), batch_size):
        yield X[b:b + batch_size], y[b:b + batch_size]

class Layer():
    def __init__(self, shape, activation = 'ReLU', dropout = 0.0):
        """
        Layer initialization.

        Parameters
        ----------
        shape : tuple
            Tuple describing the shape of the weight matrix.
        activation : string, optional
            String to describe activation. The default is 'ReLU'.
        dropout : float, optional
            Dropout regularization probability. The default is 0.0.

        Returns
        -------
        None.

        """
        self.n = shape[0]
        self.weights = np.random.randn(*shape)*np.sqrt(2/(shape[0] + shape[1]))
        self.z = None
        self.a = None
        self.d = None
        self.activation = globals()[activation]
        self.activation_d = globals()[activation + '_d']
        self.gradient = None
        self.velocity = np.zeros(shape)
        self.dropout = dropout
        
        
class Net():
    def __init__(self, input_size, loss = 'MSE', lr = 0.1, momentum = 0.0):
        """
        Initialization of Multi-Layer Perceptron network.

        Parameters
        ----------
        input_size : int
            Length of the input array.
        loss : string, optional
            String to choose loss function. The default is 'MSE'.
        lr : float, optional
            Learning rate. The default is 0.1.
        momentum : float, optional
            Momentum rate. The default is 0.0.

        Returns
        -------
        None.

        """
        self.input_size = input_size
        self.loss = globals()[loss]
        self.loss_d = globals()[loss + '_d']
        self.lr = lr
        self.momentum = momentum
        self.layers = []
        self.loss_value = 0
        self.output = None
        
    def add_layer(self, n, activation = 'ReLU', dropout = 0.0):
        """
        Method to add layer to the network.

        Parameters
        ----------
        n : int
            Number of units in this layer.
        activation : string, optional
            Activation function to be used in this layer. The default is 'ReLU'.
        dropout : float, optional
            Dropout regularization probability. The default is 0.0.

        Returns
        -------
        Layer
            The layer that was just added.

        """
        if not self.layers:
            shape = (n, self.input_size + 1)
        else:
            shape = (n, self.layers[-1].n + 1)
        self.layers.append(Layer(shape, activation = activation, dropout = dropout))
        return self.layers[-1]
        
    def feed_forward(self, X, use_dropout = False):
        """
        Take inputs and produce output of the network, using feed forward.

        Parameters
        ----------
        X : Numpy Array or Pandas DataFrame
            Input data.
        use_dropout : bool, optional
            Bool to choose if the layer should use dropout regularization during training. The default is False.

        Returns
        -------
        Numpy Array
            The output vectors for each of the input samples.

        """
        #Compute z and a for each layer
        for i, l in enumerate(self.layers):
            #First layers uses input plus a bias term
            if i == 0:
                l.z = l.weights @ bias_transform(X)
            #Other layers use a of previous layer plus a bias term
            else:
                l.z = l.weights @ bias(self.layers[i - 1].a)
            #Compute the output of the activation function    
            l.a = l.activation(l.z)
            
            #Apply dropout
            if use_dropout:
                mask = np.random.random(l.a.shape) < (1 - l.dropout)
                l.a = l.a * mask
                  
        return l.a
            
    def backprop(self, X, y):
        """
        Do backpropagation given samples and their correspodning outputs.

        Parameters
        ----------
        X : Numpy array or Pandas Dataframe
            Input data.
        y : Numpy array or Pandas Dataframe
            Target values for input data.

        Returns
        -------
        list
            The elements from the gradient corresponding to each layer.

        """
        #Go through the layers in reverse order, the index from enumeration is also reversed for simplicity
        for i, l in reversed(list(enumerate(self.layers))):
            #Last layer in network
            if i == len(self.layers) - 1:
                #Compute the loss, delta and gradient for last layer
                self.loss_value = self.loss(y, l.a.reshape(*y.shape))
                l.d = l.activation_d(l.z) * self.loss_d(y.reshape(l.a.shape), l.a)
                l.gradient = l.d.reshape(l.d.shape[0], -1, 1) @ bias(self.layers[i - 1].a).reshape(l.d.shape[0], 1, -1)
            #Hidden layers
            else:
                #Find delta for hidden layer
                l.d = (self.layers[i + 1].weights[:,1:].T @ self.layers[i + 1].d) * self.layers[i].activation_d(l.z)
                
                #Compute gradient given delta for hidden layer
                if i == 0:
                    l.gradient = l.d.reshape(l.d.shape[0], -1, 1) @ bias_transform(X).reshape(l.d.shape[0], 1, -1)
                else:
                    l.gradient = l.d.reshape(l.d.shape[0], -1, 1) @ bias(self.layers[i - 1].a).reshape(l.d.shape[0], 1, -1)
            
            #Get the average gradient of batch and reshape to size of weight matrix
            l.gradient = np.mean(l.gradient, axis = 0).reshape(*l.weights.shape)
        return [k.gradient for k in self.layers]
                
    def update_weights(self):
        """
        Update the weights after calling the feed forward and backprop methods.

        Returns
        -------
        list
            List of new weights for each layer.

        """
        for l in self.layers:
            v = self.momentum * l.velocity + self.lr * l.gradient
            l.weights -= v
            l.velocity = v
            #l.gradient *= 0
        return [k.weights for k in self.layers]          
    
    def fit(self, X, y, batch_size = 1, epochs = 100, early_stopping = True, val_size = 0.3, patience = 3, min_delta = 0.01):
        """
        Method to fit this network.

        Parameters
        ----------
        X : Numpy Array
            Input data, can be multiple samples.
        y : Numpy Array
            Corresponding target values for each input sample.
        batch_size : int, optional
            Mini-batch size, can set to 1 for SGD. The default is 1.
        epochs : int, optional
            Number of epochs to train for. The default is 100.
        early_stopping : bool, optional
            Use early stopping to prevent overfitting. The default is True.
        val_size : float, optional
            Size of validation set. The default is 0.3.
        patience : int, optional
            After this many epochs of improvement < min_delta then stop training. The default is 3.
        min_delta : TYPE, optional
            How much the loss function should improve (in percent) to prevent early stopping. The default is 0.01.

        Returns
        -------
        None.

        """
        #Setup for Early Stopping to prevent overfitting
        if early_stopping:
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = val_size)
        else:
            X_train = X
            y_train = y
        prev_loss = 1e10
        patience_counter = 0
        for epoch in range(1, epochs + 1):
            train_losses = []
            batches = make_batches(X_train, y_train, batch_size)
            for Xb, yb in batches:
                self.feed_forward(Xb, use_dropout = True)
                train_losses.append(self.loss_value)
                self.backprop(Xb, yb)
                self.update_weights()
            train_loss = np.mean(train_losses)
            print(f'Epoch {epoch}: training set loss = {round(train_loss, 3)}')
            

            #Check for early stopping
            if early_stopping:
                self.feed_forward(X_val, use_dropout = False)
                val_loss = self.loss_value
                improvement = (prev_loss - val_loss)/prev_loss
                print(f'Validation set loss = {round(val_loss, 3)}, Loss imporvement = {round(100 * improvement, 3)}%')
                if improvement <= min_delta:
                    patience_counter += 1
                else:
                    patience_counter = 0
                prev_loss = val_loss
                if patience_counter == patience:
                    break
                
                    
    def predict(self, X):
        """
        Produce outputs using the trained network given a set of input data.

        Parameters
        ----------
        X : Numpy array
            Input data.

        Returns
        -------
        Numpy array
            Predictions for each sample.

        """
        pred = self.feed_forward(X, use_dropout = False)
        return pred.reshape(X.shape[0], -1)
        