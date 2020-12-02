import tensorflow as tf
import numpy as np
import neuralnet as nn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data(path="mnist.npz")
X_old = X_test.copy()
X_train = X_train/255.0
X_test = X_test/255.0
X_train = X_train.reshape(len(X_train), -1)
X_test = X_test.reshape(len(X_test), -1)

#One-Hot Encoding
numbers = np.arange(10)
temp = [(k == numbers).astype(int) for k in y_train]
y_train = np.vstack(temp)
temp = [(k == numbers).astype(int) for k in y_test]
y_test = np.vstack(temp)

#Create a simple MLP     
net = nn.Net(input_size = 784, alpha = 0.05, mu = 0.05, loss = 'CategoricalCrossEntropy')
net.add_layer(512)
net.add_layer(256)
net.add_layer(10, activation = 'sigmoid')

#Train the network using the training data
net.fit(X_train, y_train, batch_size = 32, epochs = 10, patience = 2)

#Predict the class of the test images
pred = net.predict(X_test)

#Go from one-hot encoding back to integers between 0 and 9
y_pred = [np.argmax(r) for r in pred]
y_true = [np.argmax(r) for r in y_test]

#Compute the accuracy of the network on the test data
print(accuracy_score(y_true, y_pred))
conf_matrix = confusion_matrix(y_true, y_pred)
sns.heatmap(conf_matrix, annot = True)
plt.xlabel('Predicted')
plt.ylabel('True')
print(classification_report(y_true, y_pred))