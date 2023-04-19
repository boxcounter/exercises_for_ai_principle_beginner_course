import shopping_data
import numpy as np
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import Dense
from keras.layers import LSTM
# from keras.layers import Flatten
from keras_preprocessing import sequence

X_train, Y_train, X_test, Y_test = shopping_data.load_data()
print(f"shape(X_train) = {np.shape(X_train)}, shape(Y_train) = {np.shape(Y_train)}")
print(f"shape(X_test) = {np.shape(X_test)}, shape(Y_test) = {np.shape(Y_test)}")

vocabulary_size, word_index = shopping_data.createWordIndex(X_train, X_test)

X_train_index = shopping_data.word2Index(X_train, word_index)
X_test_index = shopping_data.word2Index(X_test, word_index)

max_len = 50
X_train_index = sequence.pad_sequences(X_train_index, maxlen=max_len)
X_test_index = sequence.pad_sequences(X_test_index, maxlen=max_len)

print(X_train_index)

features_count = 300
model = Sequential()
model.add(Embedding(
    input_dim=vocabulary_size, 
    output_dim=features_count, 
    trainable=True,
    input_length=max_len))
model.add(LSTM(units=64, return_sequences=True))
model.add(LSTM(units=64))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train_index, Y_train, batch_size=64, epochs=10)

loss, accuracy = model.evaluate(X_test_index, Y_test)
print(f"On test set: loss = {loss}, accuracy = {accuracy}")