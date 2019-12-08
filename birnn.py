#https://github.com/iamved/IMDB-sentiment-analysis/blob/master/IMDB_Sentiment_Analysis.ipynb
#https://github.com/balag59/imdb-sentiment-bidirectional-LSTM/blob/master/imdb_bilstm_train.py

from tensorflow.python.keras.datasets import imdb
from tensorflow.python.keras.preprocessing import sequence
from keras import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
import numpy as np

max_features = 10000
max_length = 500

(train_data, train_label), (test_data, test_labels) = imdb.load_data(num_words=max_features)

train_data = sequence.pad_sequences(train_data, maxlen=max_length)
test_data = sequence.pad_sequences(test_data, maxlen=max_length)

train_data = np.concatenate((train_data, test_data[:15000]))
test_data = test_data[15000:]
train_label = np.concatenate((train_label, test_labels[:15000]))
test_labels = test_labels[15000:]

embedding_size = 32
network = Sequential()
network.add(Embedding(max_features, embedding_size, input_length=max_length))
network.add(Bidirectional(LSTM(64)))
network.add(Dropout(0.2))
network.add(Bidirectional(LSTM(64)))
network.add(Dropout(0.2))
network.add(Dense(20, activation='relu'))
network.add(Dense(1, activation='sigmoid'))

network.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
history = network.fit(train_data, train_label, epochs=15, batch_size=64, validation_split=0.2)

import matplotlib.pyplot as plt
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values) +1)
plt.plot(epochs, loss_values, 'bo', label='Training Loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss values')
plt.xlabel('Epochs')
plt.ylabel('Epochs')
plt.legend()
plt.show()

plt.clf()
acc = history_dict['acc']
val_acc = history_dict['val_acc']
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()

print(network.evaluate(test_data, test_labels))
