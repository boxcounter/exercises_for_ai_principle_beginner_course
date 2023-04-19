import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import AveragePooling2D
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.utils import to_categorical

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
print(f"shape(X_train) = {np.shape(X_train)}, shape(Y_train) = {np.shape(Y_train)}")
print(f"shape(X_test) = {np.shape(X_test)}, shape(Y_test) = {np.shape(Y_test)}")

trainset_size, width, height = np.shape(X_train)
testset_size, w, h = np.shape(X_test)
assert width == w, height == h

channel_count = 1 # grayscale image has only 1 channel
X_train_with_channel = np.reshape(X_train, (trainset_size,  width, height, channel_count)) / 255
X_test_with_channel = np.reshape(X_test, (testset_size, w, h, channel_count)) / 255 

Y_train_categorical = to_categorical(Y_train, 10)
Y_test_categorical = to_categorical(Y_test, 10)

model = Sequential()

# Convolution
model.add(Conv2D(filters=6, 
                 kernel_size=(5,5), 
                 strides=(1,1), 
                 padding="valid",
                 input_shape=(width, height, channel_count),
                 activation="relu"))
model.add(AveragePooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=6, 
                 kernel_size=(5,5), 
                 strides=(1,1), 
                 padding="valid",
                 activation="relu"))
model.add(AveragePooling2D(pool_size=(2)))
model.add(Flatten())

# Full connection neuron network
model.add(Dense(units=120, activation="relu"))
model.add(Dense(units=84, activation="relu"))
model.add(Dense(units=10, activation="softmax"))
model.compile(optimizer=SGD(), loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(X_train_with_channel, Y_train_categorical, batch_size=64, epochs=100)

# Evaluate
loss, accuracy = model.evaluate(X_test_with_channel, Y_test_categorical)
print(f"On test set: loss = {loss}, accuracy = {accuracy}")