from keras.models import Sequential
import numpy as np
import matplotlib.pyplot as plt
from keras import optimizers
from keras.layers import  Dense
from keras import losses
from keras import metrics
import pickle
from RBMs.RBM_Tf import DBM

data = pickle.load(open("./_tmp/imdb.p", "rb"))

nr_nodes_list = [40, 40, 40, 40, 40]

nr_nodes_list1 = [8, 16, 24, 32, 64]

color_list = ['b', 'r', 'y', 'm', '#000000']

history_dict = []

energy_gap = []

models = []

discrimntive_epochs = 25


def run_dbm(nr_nodes, nr_nodes1, stop_at=100):
    d = DBM(layers=[[10000, nr_nodes], [nr_nodes, nr_nodes1]],
            epochs=1,
            lrate=0.1,
            batch_size=100,
            train_data=np.array(data['x_p_train']),
            val_data=np.array(data['x_val']),
            stop_at=[stop_at, stop_at],
            dump_weight_path='./_tmp/compare_DBM/' + 'H0_' + str(nr_nodes) + '_H1' + str(nr_nodes1)
            )
    d.run(1)


def run_dbm_min_energy(nr_nodes, nr_nodes1, stop_at1, stop_at2):
    d = DBM(layers=[[10000, nr_nodes], [nr_nodes, nr_nodes1]],
            epochs=1,
            lrate=0.1,
            batch_size=100,
            train_data=np.array(data['x_p_train']),
            val_data=np.array(data['x_val']),
            stop_at=[stop_at1, stop_at2],
            dump_weight_path='./_tmp/compare_DBM/MinE_H0_' + str(nr_nodes) + '_H1_' + str(nr_nodes1)
            )
    d.run(1)


def run_discriminitve_train(w1, b1, w2, b2, nr_nodes, nr_nodes1):
    model = Sequential()
    model.add(Dense(nr_nodes, activation='sigmoid', input_shape=(10000,), weights=[w1, b1]))
    model.add(Dense(nr_nodes1, activation='sigmoid', weights=[w2, b2]))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=optimizers.RMSprop(lr=0.01),
                  loss=losses.binary_crossentropy,
                  metrics=[metrics.binary_accuracy])

    history = model.fit(data['x_p_train'],
                        data['y_p_train'],
                        epochs=discrimntive_epochs,
                        batch_size=512,
                        validation_data=(data['x_val'], data['y_val']),
                        verbose=1)
    history_dict.append(history.history)
    return model

def run():
    for N1, N2 in zip(nr_nodes_list, nr_nodes_list):
        run_dbm(N1, N2)
        tmp_energy_gap = pickle.load(open("./_tmp/compare_DBM/" 'H0_' + str(N1) + '_H1' + str(N2) + "energy_diff_cd1", "rb"))
        energy_gap.append(tmp_energy_gap)
        stop_at1 = np.argmin(tmp_energy_gap[0]) + 1
        print(" Stop At 1 = ", stop_at1)
        stop_at2 = np.argmin(tmp_energy_gap[1]) + 1
        print(" Stop At 2 = ", stop_at2)

        run_dbm_min_energy(N1, N2, stop_at1, stop_at2)

        weight_list = pickle.load(open("./_tmp/compare_DBM/MinE_H0_" + str(N1) + '_H1_' + str(N2) + "weights_cd1", "rb"))
        w1 = np.asarray(weight_list[0]["W"]).astype('float32')
        b1 = np.asarray(weight_list[0]["H_W_bias"]).astype('float32')
        w2 = np.asarray(weight_list[1]["W"]).astype('float32')
        b2 = np.asarray(weight_list[1]["H_W_bias"]).astype('float32')
        run_discriminitve_train(w1, b1, w2, b2, N1, N2)


run()
plt.title('Energy Difference Test and Validation')
plt.subplot(1, 1, 1)
plt.xlabel('Epochs')
plt.ylabel('Energy Difference')
epochs = range(1, 101)
for i,node in enumerate(nr_nodes_list):
    plt.plot(epochs, energy_gap[i][0], color_list[i], label= str(node) + " Nodes")
plt.legend()
plt.grid()
plt.savefig("./_tmp/compare_DBM/EnergyVsCd_layer_0.png")
plt.show()


plt.title('Energy Difference Test and Validation')
plt.subplot(1, 1, 1)
plt.xlabel('Epochs')
plt.ylabel('Energy Difference')
epochs = range(1, 101)
for i, Nr in enumerate(zip(nr_nodes_list, nr_nodes_list1)):
    Nr1 = Nr[0]
    Nr2 = Nr[1]
    print(np.shape(energy_gap[i][1]))
    plt.plot(epochs, energy_gap[i][1], color_list[i], label= "H0_" + str(Nr1) + "_H1_" + str(Nr2))
plt.legend()
plt.grid()
plt.savefig("./_tmp/compare_DBM/EnergyVsCd_layer_1.png")
plt.show()


epochs = range(1, discrimntive_epochs+1)
plt.title('Training and Validation loss')
plt.subplot(2, 1, 1)
plt.xlabel('Epochs')
plt.ylabel('Loss')
for i, Nr in enumerate(zip(nr_nodes_list, nr_nodes_list1)):
    Nr1 = Nr[0]
    Nr2 = Nr[1]
    plt.plot(epochs, history_dict[i]['val_loss'], color_list[i], label= 'H0_'+ str(Nr1) + '_H1_' + str(Nr2))
plt.legend()
plt.grid()
plt.subplot(2, 1, 2)
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.savefig("./_tmp/compare_DBM/RBM_IMDB.png")
plt.show()


for i, node in enumerate(nr_nodes_list):
    print(models[i].evaluate(data['x_test'], data['y_test']))
