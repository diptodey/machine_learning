from keras.models import Sequential
import numpy as np
import matplotlib.pyplot as plt
from keras import optimizers
from keras.layers import  Dense
from keras import losses
from keras import metrics
import pickle
from RBMs.RBM_Tf import DBM
from keras import regularizers

batch_size = 32
epochs = 30
NrNodes = 200
data = pickle.load(open("./_tmp/imdb.p", "rb"))
data_sz = 64


trX = data['x_p_train'][:data_sz]
trY = data['y_p_train'][:data_sz]


def init_dbm():
    d = DBM(layers =[[10000,NrNodes],[NrNodes,NrNodes]],
            epochs =1,
            lrate = 0.1,
            batch_size =100,
            train_data= np.array(data['x_p_train']),
            val_data= np.array(data['x_val']),
            stop_at = [15, 15],
            dump_weight_path='./_tmp/' + str(NrNodes) + '/'
        )

    d.run(1)
    d.plot_energy_diff(0, True, False, False)
    d.plot_energy_diff(1, True, False, False)
    return d.weight_list


weight_list = pickle.load(open("./_tmp/40/weights_cd1", "rb"))


w1 = np.asarray(weight_list[0]["W"]).astype('float32')
b1 = np.asarray(weight_list[0]["H_W_bias"]).astype('float32')
w2 = np.asarray(weight_list[1]["W"]).astype('float32')
b2 = np.asarray(weight_list[1]["H_W_bias"]).astype('float32')

model = Sequential()
model.add(Dense(NrNodes, activation='sigmoid', input_shape=(10000,)))
model.add(Dense(NrNodes, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=optimizers.RMSprop(lr=0.01),
              loss=losses.binary_crossentropy,
              metrics=[metrics.binary_accuracy])

history = model.fit(trX,
                    trY,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(data['x_val'], data['y_val']),
                    verbose=1)
history_dict = history.history
val_loss_values = history_dict['val_loss']
loss_values = history_dict['loss']

model_1 = Sequential()
model_1.add(Dense(NrNodes, activation='sigmoid', input_shape=(10000,),  weights = [w1, b1]))
model_1.add(Dense(NrNodes, activation='sigmoid'))
model_1.add(Dense(1, activation='sigmoid'))

model_1.compile(optimizer=optimizers.RMSprop(lr=0.01),
                loss=losses.binary_crossentropy,
                metrics=[metrics.binary_accuracy])

history = model_1.fit(trX,
                      trY,
                      epochs=epochs,
                      batch_size=batch_size,
                      validation_data=(data['x_val'], data['y_val']),
                      verbose=1)
history_dict_1 = history.history
val_loss_values_1 = history_dict_1['val_loss']
loss_values_1 = history_dict_1['loss']

model_2 = Sequential()
model_2.add(Dense(NrNodes,activation='sigmoid', input_shape=(10000,), weights = [w1, b1]))
model_2.add(Dense(NrNodes, activation='sigmoid', weights = [w2, b2]))
model_2.add(Dense(1, activation='sigmoid'))

model_2.compile(optimizer=optimizers.RMSprop(lr=0.01),
                loss=losses.binary_crossentropy,
                metrics=[metrics.binary_accuracy])

history_2 = model_2.fit(trX,
                        trY,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(data['x_val'], data['y_val']),
                        verbose=1)

history_dict_2 = history_2.history
val_loss_values_2 = history_dict_2['val_loss']
loss_values_2 = history_dict_2['loss']


epochs = range(1, len(loss_values_1) + 1)
plt.title('Training and Validation loss')
plt.subplot(2, 1, 1)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.plot(epochs, val_loss_values, '#000000', label='Validation loss NN')
plt.plot(epochs, val_loss_values_1, 'b-', label='Validation loss RBM one Layer')
plt.plot(epochs, val_loss_values_2, 'r-', label=' Validation loss RBM two layer')
plt.legend()
plt.grid()

plt.subplot(2, 1, 2)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.plot(epochs, loss_values, '#000000', label='Training loss NN')
plt.plot(epochs, loss_values_1, 'b-', label='Training loss RBM one layer')
plt.plot(epochs, loss_values_2, 'r-', label=' Training loss RBM two layer')
plt.legend()
plt.grid()
plt.savefig("./_tmp/" + str(NrNodes) + "/RBM_IMDB_layers_bsz64.png")
plt.show()

print( model.evaluate(data['x_test'],  data['y_test']))
print( model_1.evaluate(data['x_test'],  data['y_test']))
print( model_2.evaluate(data['x_test'],  data['y_test']))