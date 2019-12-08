
import numpy as np

from tensorflow_core.python.keras.datasets import imdb
from tensorflow_core.python.keras.preprocessing import sequence

max_features = 10000
maxlen = 1000
batch_size = 32

print('Loading data...')
(input_train, y_train), (input_test, y_test) = imdb.load_data(num_words=max_features)
print(len(input_train), 'train sequences')
print(len(input_test), 'test sequences')

print('Pad sequences (samples x time)')
input_train = sequence.pad_sequences(input_train, maxlen=maxlen)
input_test = sequence.pad_sequences(input_test, maxlen=maxlen)
print('input_train shape:', input_train.shape)
print('input_test shape:', input_test.shape)
# make the training data 80% and testing 20%
x_train = np.concatenate((input_train, input_test[:15000]))
input_test = input_test[15000:]
y_train = np.concatenate((y_train, y_test[:15000]))
y_test = y_test[15000:]


from tensorflow_core.python.keras import models
from tensorflow_core.python.keras import layers

embedding_size = 128
model = models.Sequential()
model.add(layers.Embedding(max_features, embedding_size))
model.add(layers.Bidirectional(layers.LSTM(128)))
# model.add(layers.Dropout(0.2))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))


model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])
his = model.fit(x_train, y_train,
                    epochs=3,
                    batch_size=64,
                    validation_split=0.025)


results = model.evaluate(input_test, y_test)
print(results)

print("Done!")



