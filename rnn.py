# from github susanli2016
#https://github.com/andyngo95/SA_Positive_Negative_Comments/blob/master/Sentiment_Analysis_v2.ipynb
#https://towardsdatascience.com/light-on-math-ml-attention-with-keras-dc8dbc1fad39
#https://github.com/thushv89/attention_keras
from tensorflow.python.keras.datasets import imdb
#from attention_keras.layers.attention import AttentionLayer
import numpy as np

(train_data, train_label), (test_data, test_labels) = imdb.load_data(num_words=10000)

print("length of train_data and test_data: ", len(train_data), len(test_data))
print(train_data[0])
print(train_label[0])
print(max([max(sequence) for sequence in train_data]))

word_index = imdb.get_word_index()

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])
print("review in train_data[0]: ", decoded_review)

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, each_sentence in enumerate(sequences):
        results[i, each_sentence] = 1
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
print(x_train[0].ndim)
print(x_test.ndim)

y_train = np.asarray(train_label).astype('float32')
y_test = np.asarray(test_labels).astype('float32')
index = 2
print(x_train[index])
print(y_train[index])

from keras import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout

embedding_size = 32
model = Sequential()
model.add(Embedding(10000, embedding_size, input_length=500))
model.add(LSTM(128))
model.add(Dropout(0.4))
model.add(LSTM(128))
model.add(Dropout(0.4))
# need attention
model.add(Dense(256), activation='relu')
model.add(Dropout(0.5))
model.add(Dense(1, activation='softmax'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
x_val = x_train[:10000]
partial_x_training = x_train[10000:]
y_val = y_train[:10000]
partial_y_training = y_train[10000:]
history = model.fit(partial_x_training, partial_y_training, epochs=20, batch_size=512, validation_data=(x_val, y_val))

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
