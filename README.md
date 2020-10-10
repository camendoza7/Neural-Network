# Neural-Network
An implementation of a dense Multi-Layer Perceptron from scratch.

This code allows a user to create a MLP with some configuration such as the number of layers, the number of nodes in each layer and the activation function used in each layer.
This project only supports dense MLPs and only has a couple of activation functions and loss functions.

Activation Functions
-------------------------
* ReLU
* Sigmoid

Loss Functions
-------------------------
* MSE
* Categorical Cross Entropy

The optimization technique that is implemented is Mini-Batch Gradient Descent with momentum however, if you would like to make it Stochastic Gradient Descent you can simply set the mini-batch size = 1.

## Simple Example Classifying the MNIST Handwritten Digits Dataset
```python
import neuralnet as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#Load the MNIST handwritten digits dataset
digits = datasets.load_digits()
X = digits.images.reshape((len(digits.images), -1))
labels = digits.target

#Normalize the data between 0 and 1
normalizer = MinMaxScaler()
X = normalizer.fit_transform(X)

#One-Hot Encode the output
lr = np.arange(10)
labels = [(k == lr).astype(int) for k in labels]
labels = np.vstack(labels)   

#Split data
X_train, X_test, y_train, y_test = train_test_split(X, labels)
  
#Create a simple MLP     
net = nn.Net(input_size = 64, alpha = 0.1, mu = 0.1, loss = 'CategoricalCrossEntropy')
net.add_layer(512)
net.add_layer(56)
net.add_layer(10, activation = 'sigmoid')

#Train the network using the training data
net.fit(X_train, y_train, minibatch = 32, epochs = 10)

#Predict the class of the test images
pred = net.predict(X_test)

#Go from one-hot encoding back to integers between 0 and 9
y_pred = [np.argmax(r) for r in pred]
y_true = [np.argmax(r) for r in y_test]

#Compute the accuracy of the network on the test data
print(accuracy_score(y_true, y_pred))
```
