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
from keras.layers import  Dense, Input

#data = pickle.load(open("imdb.p", "rb"))


model = models.Sequential()
model.add(layers.Dense(16, kernel_regularizer=regularizers.l2(0.001),activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, kernel_regularizer=regularizers.l2(0.001),activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))



model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
from keras.utils import plot_model
from keras.models import Model, load_model
from keras.layers.noise import GaussianNoise
input_tensor_1         = Input(shape=(10000,))

x5 = Dense(16, activation='relu')(input_tensor_1)
x6 = GaussianNoise(stddev=0.001)(x5)
x7 = Dense(64, activation='relu')(x6)
model_hd_gnoisereg  = Model(input_tensor_1,         Dense(1, activation='sigmoid')(x7))
model_hd_gnoisereg.compile (optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])


plot_model(model_hd_gnoisereg, to_file='model_gnoise.png', show_shapes = True)

exit()
history = model.fit(data['partial_x_train'],
                    data['partial_y_train'],
                    epochs=100,
                    batch_size=512,
                    validation_data=(data['x_val'], data['y_val']))
history_dict = history.history
acc = history_dict['acc']
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

pickle.dump( history.history, open('history_l2reg001_e100.p', 'wb'))

epochs = range(1, len(acc) + 1)
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b+', label='Validation loss')

plt.savefig("l2reg_001_e100.png")
#plt.save("Training_validation_loss_orig.jpg")
plt.show()