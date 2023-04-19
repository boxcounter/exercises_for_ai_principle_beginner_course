import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.utils import to_categorical

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

print(f"shape(X_train) = {np.shape(X_train)}, shape(Y_train) = {np.shape(Y_train)}")
print(f"shape(X_test) = {np.shape(X_test)}, shape(Y_test) = {np.shape(Y_test)}")

trainset_size, width, height = np.shape(X_train)
testset_size, w, h = np.shape(X_test)

assert width == w, height == h

X_train_flatten = np.reshape(X_train, (trainset_size, width * height)) / 256 # divided by 256 to normalize
X_test_flatten = np.reshape(X_test, (testset_size, width * height)) / 256

Y_train_categorical = to_categorical(Y_train, 10)
Y_test_categorical = to_categorical(Y_test, 10)

model = Sequential()
model.add(Dense(units=128, activation='relu', input_dim=width * height))
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=SGD(), metrics=['accuracy'])
model.fit(X_train_flatten, Y_train_categorical, batch_size=64, epochs=100)

loss, accuracy = model.evaluate(X_test_flatten, Y_test_categorical)
print(f"on test set: loss = {loss}, accuracy = {accuracy}")