from keras.datasets import mnist
from mlp import MLP
import numpy as np
import keras

(xtrain, ytrain) , (xtest, ytest) = mnist.load_data()

xtrain = np.reshape(xtrain, (60000, 784))
xtest = np.reshape(xtest, (10000, 784))
ytrain = keras.utils.to_categorical(ytrain, 10)

mlp = MLP(np.shape(xtrain)[1], 512, 512, 10)

epochs = 100
n_test = np.shape(xtest)[0]

for i in range(epochs):
    mlp.train(0.001, xtrain, ytrain)
    correct = 0
    for j in range(n_test):
        pred = mlp.predict(xtest[j])
        if np.argmax(pred) == ytest[j]:
            correct += 1

    print('test accuracy: ', correct / n_test)



