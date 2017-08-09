from keras.datasets import mnist
import numpy as np
from keras.utils import to_categorical

print("Downloading data...")
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = np.reshape(X_train, (60000, 784))
# X_test = np.reshape(X_test, (10000, 784))
Y_train = to_categorical(Y_train, 10)
# Y_test = keras.utils.to_categorical(Y_test, 10)

X_train = X_train / 255
# X_test = X_test / 255

nb_classes = Y_train.shape[1]
print(nb_classes, 'classes')

dims = X_train.shape[1]
print(dims, 'dims')

from keras.models import Sequential
from keras.layers import Dense, Activation

print("Building model...")

model = Sequential()
model.add(Dense(1024, input_shape=(dims,), activation='sigmoid'))
model.add(Dense(nb_classes, input_shape=(dims,), activation='sigmoid'))

model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['acc'])
model.fit(X_train, Y_train)

from keras.models import load_model

model.save('A:/Machine Learning/MLP/my_model.h5')
