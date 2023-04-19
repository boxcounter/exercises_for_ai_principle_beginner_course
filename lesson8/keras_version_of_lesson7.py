import dataset
import numpy as np
import plot_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

dataset_size = 100
X, Y = dataset.get_beans(dataset_size)
plot_utils.show_scatter(X, Y)

model = Sequential()
dense = Dense(units=1, activation='sigmoid', input_dim=2)
model.add(dense)

model.compile(loss='mse', optimizer=SGD(learning_rate=0.1), metrics=['accuracy'])
model.fit(X, Y, epochs=500, batch_size=10)

plot_utils.show_scatter_surface(X, Y, model)