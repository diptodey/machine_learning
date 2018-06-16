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

batch_size = 128
epochs = 60
NrNodes = 40
data = pickle.load(open("./_tmp/imdb.p", "rb"))
data_sz = 128


trX = data['x_p_train'][:data_sz]
trY = data['y_p_train'][:data_sz]
print(np.shape(trX))

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
#weight_list = init_dbm()
#print(weight_list)


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
model_1.add(Dense(NrNodes, activation='sigmoid', input_shape=(10000,), weights = [w1, b1]))
model_1.add(Dense(NrNodes, activation='sigmoid', weights = [w2, b2]))
model_1.add(Dense(1, activation='sigmoid'))

model_1.compile(optimizer=optimizers.RMSprop(lr=0.01),
                loss=losses.binary_crossentropy,
                metrics=[metrics.binary_accuracy])

history_1 = model_1.fit(trX,
                        trY,
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
plt.savefig("./_tmp/" + str(NrNodes) + "/RBM_IMDB.png")
plt.show()

print( model.evaluate(data['x_test'],  data['y_test']))
print( model_1.evaluate(data['x_test'],  data['y_test']))