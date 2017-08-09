from keras.datasets import mnist
import numpy as np
from keras.utils import to_categorical


print("Downloading data...")
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()


X_test = np.reshape(X_test, (10000, 784))
X_test = X_test / 255


from keras.models import load_model

print("Loading model...")
model = load_model('A:/Machine Learning/MLP/my_model.h5')


print("Testing model...")
pred = model.predict(X_test)
result = np.argmax(pred, axis=1)
correct = np.sum(result == Y_test)

print('correct: ', correct/10000)