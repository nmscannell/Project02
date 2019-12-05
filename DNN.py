#%% Internet movie database

from tensorflow_core.python.keras.datasets import imdb
import numpy as np
(train_data, train_label), (test_data, test_labels) = imdb.load_data(num_words=10000)
# only keep the top 10,000 most frequently occurring words in the training data, 
# in other words, the words in the movie review only come from the dictionary that 
# only has these 10,000 words.  



print("length of train_data and test_data: ", len(train_data), len(test_data))

print(train_data[0])
# length of train_data[0] is 218, and this review is a list of word indices encoding 
# a sequence of words from the 10,000-word dictionary
print(train_label[0])
# 0 or 1, with 0 for negative review, and 1 for positive review

print(max([max(sequence) for sequence in train_data]))
# for each review in train_data find the max index, then find the max of 
# all max indices in all reviews

word_index = imdb.get_word_index()
# word_index is a dictionary with key being the word and value being the index
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
# build a dictionary with key being the index and value being the word in the dictionary

decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])
# decode the review, note that the indices are offset by 3 because 0, 1, and 2 are reserved 
# indices for "padding", "start of sequence", and "unknown".
print("review in train_data[0]: ", decoded_review)


# lists of integers cannot be fed into neural network, instead lists need to be turned
# into tensors by using one-hot encoding, the basic idea is to use 10,000-dimension one-hot
# vector to encode which words appear in the review that leads to the positive or negative 
# review.



def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, each_sequence in enumerate(sequences):
        # type of each_sequence is list
        results[i, each_sequence] = 1
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)


y_train = np.asarray(train_label).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

#make the training data 80% and testing 20%
x_train = np.concatenate((x_train, x_test[:15000]))
x_test = x_test[15000:]
y_train = np.concatenate((y_train, test_labels[:15000]))
y_test = y_test[15000:]

from tensorflow_core.python.keras import models
from tensorflow_core.python.keras import layers

network = models.Sequential()
network.add(layers.Dense(128, activation='relu', input_shape=(10000,)))
network.add(layers.Dense(64, activation='relu'))
network.add(layers.Dense(64, activation='relu'))
network.add(layers.Dense(32, activation='relu'))
network.add(layers.Dense(32, activation='relu'))
network.add(layers.Dense(16, activation='relu'))
network.add(layers.Dense(16, activation='relu'))
network.add(layers.Dense(8, activation='relu'))
network.add(layers.Dense(8, activation='relu'))
network.add(layers.Dense(1, activation='sigmoid'))

# like before
network.compile(optimizer='rmsprop',
                loss='binary_crossentropy',
                metrics=['acc'])


# the other for validation. The rationale is to reduce the risk of over-fitting

x_val = x_train[:1000]
partial_x_training = x_train[1000:]
y_val = y_train[:1000]
partial_y_training = y_train[1000:]

his = network.fit(partial_x_training, partial_y_training, 
                      epochs=3, batch_size=1024, validation_data=(x_val, y_val))


import matplotlib.pyplot as plt

history_dict = his.history
# loss value for training data
loss_values = history_dict['loss']
# loss value for validation data
val_loss_values = history_dict['val_loss']

epochs = range(1, len(loss_values) + 1)

# blue dots for training loss
plt.plot(epochs, loss_values, 'bo', label='Training Loss')
# solid blue line for validation loss
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
results = network.evaluate(x_test, y_test)
print(results)

print("Done!")

# # now note that you have to create a new model to have a clean slate
# network = models.Sequential()
# network.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
# network.add(layers.Dense(16, activation='relu'))
# network.add(layers.Dense(1, activation='sigmoid'))
# network.compile(optimizer='rmsprop',
#                 loss='binary_crossentropy',
#                 metrics=['acc'])
# network.fit(x_train, y_train,epochs=3, batch_size=512)

