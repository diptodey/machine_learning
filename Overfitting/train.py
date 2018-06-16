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
import pickle

batch_size = 512
epochs = 20

def get_oup_model_nornd(_model, _data):
    return np.array([X[0] for X in _model.predict(_data)])

data = pickle.load(open("imdb.p", "rb"))

Input_tensor_1   = Input(shape=(10000,))
Input_tensor_2   = Input(shape=(10000,))
Input_tensor_3   = Input(shape=(10000,))
Input_tensor_N   = Input(shape=(10000,))
Input_tensor_C   = Input(shape=(10000,))

x3 = Dense(4, activation='relu')(Input_tensor_N)
x4 = Dense(4, activation='relu')(x3)
model_N = Model(Input_tensor_N, Dense(1, activation='sigmoid')(x4))
model_N.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

def hist_plot(data, bins, facecolor, labels, span = (-0.2, 0.2)):
    fig = plt.figure(figsize=(20,10))  # create a figure object
    ax = fig.add_subplot(1, 1, 1)
    for i in range(0, len(data)):
        ax.hist(data[i], bins=bins, range = span, facecolor= facecolor[i],
                histtype='step', stacked=True, fill = False, alpha=1,
                normed=True, label = labels[i], linewidth=4.0 )
    ax.minorticks_on()
    ax.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize = 15)
    ax.tick_params(direction='out', length=10, width=10, colors='b', labelsize  = 18)
    plt.savefig("hist_4_4.png")
    plt.show()

def run_model1(trX, trY, newmodel= True):
    if not newmodel:
        # Model reconstruction from JSON file
        model = load_model('my_model_1.h5')
    else:
        x1 = Dense(4, activation='relu')(Input_tensor_1)
        x2 = Dense(4, activation='relu')(x1)
        model = Model(Input_tensor_1, Dense(1, activation='sigmoid')(x2))
        model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(trX, trY, epochs=epochs, batch_size=batch_size)
    Z = get_oup_model_nornd(model, data['x_p_train_2'])
    # Save the model architecture
    model.save('my_model_1.h5')
    del(model)
    return Z

def run_model2(trX, trY):
    history_N = model_N.fit(trX, trY, epochs=epochs, batch_size=batch_size, validation_data=(data['x_val'], data['y_val']))
    history_dict = history_N.history
    return history_dict

val_loss_values_n = []
loss_values_n = []

trY = run_model1(data['x_p_train_1'], data['y_p_train_1'])
history_dict = run_model2(data['x_p_train_2'], trY)
val_loss_values_n = history_dict['val_loss']
loss_values_n = history_dict['loss']

"""
trY = run_model1(data['x_p_train_1'], data['y_p_train_1'])
history_dict = run_model2(data['x_val'], trY)
val_loss_values_n =  val_loss_values_n + history_dict['val_loss']
loss_values_n = loss_values_n + history_dict['loss']

trY = run_model1(data['x_p_train_1'], data['y_p_train_1'])
history_dict = run_model2(data['x_val'], trY)
val_loss_values_n =  val_loss_values_n + history_dict['val_loss']
loss_values_n = loss_values_n + history_dict['loss']
"""
x5 = Dense(4, activation='relu')(Input_tensor_C)
x6 = Dense(4, activation='relu')(x5)
model_C = Model(Input_tensor_C, Dense(1, activation='sigmoid')(x6))
model_C.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
history_C = model_C.fit(data['x_train'], data['y_train'], epochs=epochs, batch_size=512, validation_data=(data['x_val'], data['y_val']))
history_dict = history_C.history
val_loss_values_c = history_dict['val_loss']
loss_values_c = history_dict['loss']


epochs = range(1, len(loss_values_n) + 1)
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.plot(epochs, loss_values_n, 'b-', label='N Training loss')
plt.plot(epochs, val_loss_values_n, 'b-+', label=' N Validation loss')
epochs = range(1, len(loss_values_c) + 1)
plt.plot(epochs, loss_values_c, 'r-', label='C Training loss')
plt.plot(epochs, val_loss_values_c, 'r-+', label=' C Validation loss')
plt.legend()

plt.savefig("4_4.png")
plt.show()


print( model_N.evaluate(data['x_test'],  data['y_test']))
print( model_C.evaluate(data['x_test'],  data['y_test']))

model = load_model('my_model_1.h5')
weights, biases = model.layers[1].get_weights()

weights_c, biases_c = model_C.layers[1].get_weights()
hist_plot([ weights.flatten(),  weights_c.flatten()],
          50,
          ['y', 'b', 'black', 'steelblue'],
          ['Model', 'C Model'])
del(model_N)
del(model_C)
