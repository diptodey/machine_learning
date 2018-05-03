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
from RBMs.RBM_Tf import DBM

batch_size = 512
epochs = 20
data = pickle.load(open("imdb.p", "rb"))


d = DBM(layers =[[10000,16],[16,16]],
        epochs =1,
        lrate = 0.1,
        batch_size =100,
        train_data= np.array(data['x_p_train']),
        val_data= np.array(data['x_val']),
        stop_at = [20, 40],
        dump_weight_path = 'weight_4_4.p'
    )

print(d.weight_list)


#weight_list = pickle.load(open("weight_16_16.p", "rb"))
weight_list = d.weight_list
#print(weight_list)


data = pickle.load(open("imdb.p", "rb"));
w1 = np.asarray(weight_list[0]["W"]).astype('float32')
b1 = np.asarray(weight_list[0]["H_W_bias"]).astype('float32')
w2 = np.asarray(weight_list[1]["W"]).astype('float32')
b2 = np.asarray(weight_list[1]["H_W_bias"]).astype('float32')

model = Sequential()
model.add(Dense(16, activation='sigmoid', input_shape=(10000,)))
model.add(Dense(16, activation='sigmoid'))
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
model_1.add(Dense(16, activation='sigmoid', input_shape=(10000,), weights = [w1, b1]))
model_1.add(Dense(16, activation='sigmoid'))
model_1.add(Dense(1, activation='sigmoid'))

model_1.compile(optimizer=optimizers.RMSprop(lr=0.1),
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

epochs = range(1, len(loss_values_1) + 1)
plt.title('Training and Validation loss')
plt.subplot(2, 1, 1)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.plot(epochs, val_loss_values, 'b-', label='Validation loss')
plt.plot(epochs, val_loss_values_1, 'r-', label=' Validation loss RBM ')
plt.legend()
plt.grid()

plt.subplot(2, 1, 2)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.plot(epochs, loss_values, 'b-', label='Training loss')
plt.plot(epochs, loss_values_1, 'r-', label=' Training loss RBM')
plt.legend()
plt.grid()
plt.savefig("RBM_IMDB.png")
plt.show()

print( model.evaluate(data['x_test'],  data['y_test']))
print( model_1.evaluate(data['x_test'],  data['y_test']))