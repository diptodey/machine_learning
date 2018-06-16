from keras.models import Sequential
import numpy as np
from keras import models
from keras import layers
from keras import regularizers
from keras.datasets import imdb
import matplotlib.pyplot as plt
from keras import losses
from keras import optimizers
import pickle

"""
IMDB dataset: a set of 50,000 highly polarized reviews from the
Internet Movie Database. They’re split into 25,000 reviews for training and 25,000
reviews for testing, each set consisting of 50% negative and 50% positive reviews.

The argument num_words=10000 means you’ll only keep the top 10,000 most frequently
occurring words in the training data. Rare words will be discarded. This allows
you to work with vector data of manageable size.
"""
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

x_val = x_train[:10000]
y_val = y_train[:10000]
partial_x_train = x_train[10000:]
partial_y_train = y_train[10000:]

partial_y_train_unlearn = np.random.randint(2,size = np.shape(partial_y_train))*partial_y_train



plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

model = models.Sequential()
model.add(layers.Dense(16, kernel_regularizer=regularizers.l2(0.001),activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, kernel_regularizer=regularizers.l2(0.001),activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))


model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))
history_dict = history.history
acc = history_dict['acc']
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

#plt.hold()


model_unlearn = models.Sequential()
model_unlearn.add(layers.Dense(16, kernel_regularizer=regularizers.l2(0.001),activation='relu', input_shape=(10000,)))
model_unlearn.add(layers.Dense(16, kernel_regularizer=regularizers.l2(0.001),activation='relu'))
model_unlearn.add(layers.Dense(1, activation='sigmoid'))

model_unlearn.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history1 = model_unlearn.fit(partial_x_train,
                    partial_y_train,
                    epochs=6,
                    batch_size=512,
                    validation_data=(x_val, y_val))


history2 = model_unlearn.fit(partial_x_train,
                             partial_y_train_unlearn,
                             epochs=2,
                             batch_size=512,
                             validation_data=(x_val, y_val))


history3 = model_unlearn.fit(partial_x_train,
                             partial_y_train,
                             epochs=6,
                             batch_size=512,
                             validation_data=(x_val, y_val))


history4 = model_unlearn.fit(partial_x_train,
                    partial_y_train_unlearn,
                    epochs=2,
                    batch_size=512,
                    validation_data=(x_val, y_val))


history5 = model_unlearn.fit(partial_x_train,
                    partial_y_train,
                    epochs=8,
                    batch_size=512,
                    validation_data=(x_val, y_val))



history_dict_unlearn = {}

for key in history1.history.keys():
    history_dict_unlearn[key] = history1.history[key] +\
                                history3.history[key] + history5.history[key]

"""
model_antilearn = models.Sequential()
model_antilearn.add(layers.Dense(4, activation='relu', input_shape=(10000,)))
model_antilearn.add(layers.Dense(4, activation='relu'))
model_antilearn.add(layers.Dense(1, activation='sigmoid'))

model_antilearn.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history1 = model_antilearn.fit(partial_x_train,
                    partial_y_train,
                    epochs=6,
                    batch_size=512,
                    validation_data=(x_val, y_val))


history2 = model_antilearn.fit(partial_x_train,
                             -partial_y_train,
                             epochs=2,
                             batch_size=512,
                             validation_data=(x_val, y_val))


history3 = model_antilearn.fit(partial_x_train,
                    partial_y_train,
                    epochs=14,
                    batch_size=512,
                    validation_data=(x_val, y_val))



history_dict_antilearn = {}

for key in history1.history.keys():
    history_dict_antilearn[key] = history1.history[key] +\
                                history3.history[key]

"""
unlearn_acc = history_dict_unlearn['acc']
unlearn_loss_values = history_dict_unlearn['loss']
unlearn_val_loss_values = history_dict_unlearn['val_loss']

"""
antilearn_acc = history_dict_antilearn['acc']
antilearn_loss_values = history_dict_antilearn['loss']
antilearn_val_loss_values = history_dict_antilearn['val_loss']
"""
epochs = range(1, len(unlearn_acc) + 1)
plt.plot(epochs, unlearn_loss_values, 'ro', label='Training loss')
plt.plot(epochs, unlearn_val_loss_values, 'r', label='Validation loss')
#plt.plot(epochs, antilearn_loss_values, 'go', label='Training loss')
#plt.plot(epochs, antilearn_val_loss_values, 'g', label='Validation loss')
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')

plt.savefig("Training_validation_loss.png")
#plt.save("Training_validation_loss_orig.jpg")
plt.show()



