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


data = pickle.load(open("imdb.p", "rb"))


model = models.Sequential()
model.add(layers.Dense(4,activation='sigmoid', input_shape=(10000,)))
model.add(layers.Dense(4,activation='sigmoid'))
model.add(layers.Dense(1, activation='sigmoid'))


model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history_list =[]

for i in range(0,10):
    _history = model.fit(data['partial_x_train'],
                        data['partial_y_train'],
                        epochs=5,
                        batch_size=512,
                        validation_data=(data['x_val'], data['y_val']))
    history_list.append(_history.history)

    if(i <9):
        model.fit(data['partial_x_train'],
                  data['partial_y_train'],
                  epochs=2,
                  batch_size=512,
                  validation_data=(data['x_val'], data['y_val']))

    _history = model.fit(data['partial_x_train'],
                        data['partial_y_train'],
                        epochs=5,
                        batch_size=512,
                        validation_data=(data['x_val'], data['y_val']))
    history_list.append(_history.history)

history_dict_unlearn = {'acc':[], 'val_acc': [],'loss': [], 'val_loss': []}

for item in history_list:
    for key in item.keys():
        history_dict_unlearn[key] += item[key]

acc = history_dict_unlearn['acc']
loss_values = history_dict_unlearn['loss']
val_loss_values = history_dict_unlearn['val_loss']

pickle.dump( history_dict_unlearn, open('history_unlearn_2_10_e100.p', 'wb'))

epochs = range(1, len(acc) + 1)
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b+', label='Validation loss')

plt.savefig("unlearn_e100.png")
#plt.save("Training_validation_loss_orig.jpg")
plt.show()

