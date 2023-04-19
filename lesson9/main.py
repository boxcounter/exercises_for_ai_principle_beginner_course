import dataset
import plot_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

X, Y = dataset.get_beans(100)
plot_utils.show_scatter(X, Y)

model = Sequential()
model.add(Dense(units=8, activation='relu', input_dim=2))
model.add(Dense(units=8, activation='relu'))
model.add(Dense(units=8, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(optimizer=SGD(learning_rate=0.1), loss='mse', metrics=['accuracy'])
model.fit(X, Y, epochs=5000, batch_size=10)

plot_utils.show_scatter_surface_with_model(X, Y, model)