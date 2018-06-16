from keras.models import Sequential
import numpy as np
from keras.models import Model
from keras.models import load_model
from keras import layers
from keras import regularizers
from keras.datasets import imdb
import matplotlib.pyplot as plt
from keras import losses
from keras import optimizers
from keras.layers import  Dense, Input, Add, concatenate
from keras import losses
from keras import metrics
import pickle

batch_size = 512
epochs = 20

weights = pickle.load(open("weights.p", "rb"));
data = pickle.load(open("imdb.p", "rb"));
w1 = np.asarray(weights["inp_hid"]["W"]).astype('float32')
b1 = np.asarray(weights["inp_hid"]["H_W_bias"]).astype('float32')
w2 = np.asarray(weights["hid_oup"]["W"]).astype('float32')
b2 = np.asarray(weights["hid_oup"]["H_W_bias"]).astype('float32')
model = Sequential()
model.add(Dense(8, activation='sigmoid', input_shape=(10000,), weights = [w1, b1]))
model.add(Dense(8, activation='sigmoid',weights = [w2, b2] ))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=optimizers.RMSprop(lr=0.01),
              loss=losses.binary_crossentropy,
              metrics=[metrics.binary_accuracy])

history = model.fit(data['x_p_train'],
                      data['y_p_train'],
                      epochs=epochs,
                      batch_size=batch_size,
                      validation_data=(data['x_val'], data['y_val']),
                    verbose=1)
history_dict = history.history
val_loss_values = history_dict['val_loss']
loss_values = history_dict['loss']

model_1 = Sequential()
model_1.add(Dense(8, activation='sigmoid', input_shape=(10000,), weights = [w1, b1]))
model_1.add(Dense(8, activation='sigmoid',weights = [w2, b2] ))
model_1.add(Dense(1, activation='sigmoid'))

model_1.compile(optimizer=optimizers.RMSprop(lr=0.01),
              loss=losses.binary_crossentropy,
              metrics=[metrics.binary_accuracy])

history_1 = model_1.fit(data['x_p_train'],
                      data['y_p_train'],
                      epochs=epochs,
                      batch_size=batch_size,
                      validation_data=(data['x_val'], data['y_val']),
                    verbose=1)

history_dict_1 = history_1.history
val_loss_values_1 = history_dict_1['val_loss']
loss_values_1 = history_dict_1['loss']

for i,layer in enumerate(model_1.layers):
    if (i < 2):
        layer.trainable = False

model_1.compile(optimizer=optimizers.RMSprop(lr=0.01),
              loss=losses.binary_crossentropy,
              metrics=[metrics.binary_accuracy])

history_2 = model_1.fit(data['x_p_train'],
                      data['y_p_train'],
                      epochs=epochs,
                      batch_size=batch_size,
                      validation_data=(data['x_val'], data['y_val']),
                    verbose=1)

history_dict_2 = history_2.history
val_loss_values_2 = history_dict_2['val_loss']
loss_values_2 = history_dict_2['loss']

epochs = range(1, len(loss_values) + 1)
plt.title('Training and Validation loss')
plt.subplot(2, 1, 1)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.plot(epochs, val_loss_values, 'b-', label='Validation loss')
plt.plot(epochs, val_loss_values_1, 'r-', label=' Validation loss RBM ')
plt.plot(epochs, val_loss_values_2, 'g-', label=' Validation loss RBM init weight freeze')
plt.legend()
plt.grid()

plt.subplot(2, 2, 1)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.plot(epochs, loss_values, 'b-', label='Training loss')
plt.plot(epochs, loss_values_1, 'r-', label=' Training loss RBM')
plt.plot(epochs, loss_values_2, 'g-', label=' Training loss RBM init weight freeze')
plt.legend()
plt.grid()
plt.savefig("RBM_IMDB.png")
plt.show()

print( model.evaluate(data['x_test'],  data['y_test']))
