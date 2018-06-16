import numpy as np
from keras.models import Model
from keras.layers import  Dense, Input, concatenate
from keras import regularizers
from keras.datasets import imdb
import matplotlib.pyplot as plt
from keras import losses
from keras import optimizers
import pickle


def get_oup_model(_model, _data):
    return np.array([round(X[0]) for X in _model.predict(_data)])

def get_oup_model_nornd(_model, _data):
    return np.array([X[0] for X in _model.predict(_data)])

def append_data_col(_col, _data):
    shape_inp = _data.shape
    X = np.zeros([shape_inp[0], shape_inp[1] + 1])
    X[:, 0] = _col
    X[:, 1:] = _data
    return  X

def append_2data_col(_col1, _col2, _data):
    shape_inp = _data.shape
    X = np.zeros([shape_inp[0], shape_inp[1] + 2])
    X[:, 0] = _col1
    X[:, 1] = _col2
    X[:, 2:] = _data
    return  X

MODEL_EPOCHS = 10
MODEL_C1_EPOCHS = 50
MODEL_C2_EPOCHS = 50
MODEL_COMB_EPOCHS = 50

MODEL_BATCH_SZ = 512

data = pickle.load(open("imdb_causal.p", "rb"))
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
sgd1 = optimizers.SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)

fig = plt.figure(figsize=(20,10))  # create a figure object
fig.suptitle('Prediction Accuracy', fontsize=12)
ax = fig.add_subplot(1, 1, 1)
ax.plot


input_tensor         = Input(shape=(10000,))
input_tensor_causal1 = Input(shape=(10001,))
input_tensor_causal2 = Input(shape=(10002,))
input_tensor_comb    = Input(shape=(3,))

# Non Causal Layers
x1 = Dense(4, activation='relu')(input_tensor)
x2 = Dense(4, activation='relu')(x1)
#Causal 1 Neunet
x3 = Dense(10, activation='relu')(input_tensor_causal1)
x4 = Dense(10, activation='relu')(x3)
#Causal2 NeuNet
x5 = Dense(6, activation='relu')(input_tensor_causal2)
x6 = Dense(6, activation='relu')(x5)
#Comb Neunet
x7 = Dense(4, activation='sigmoid')(input_tensor_comb)
x8 = Dense(4, activation='sigmoid')(x7)

model           = Model(input_tensor,         Dense(1, activation='sigmoid')(x2))
model_causal_1  = Model(input_tensor_causal1, Dense(1, activation='sigmoid')(x4))
model_causal_2  = Model(input_tensor_causal2, Dense(1, activation='sigmoid')(x6))
model_comb      = Model(input_tensor_comb,    Dense(1, activation='sigmoid')(x8))

model.compile         (optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
model_causal_1.compile(optimizer= sgd1, loss='binary_crossentropy', metrics=['accuracy'])
model_causal_2.compile(optimizer= sgd1, loss='binary_crossentropy', metrics=['accuracy'])
model_comb.compile    (optimizer= sgd, loss='binary_crossentropy', metrics=['accuracy'])



trX = data['partial_x_train_1']
trY = data['partial_y_train_1']
history = model.fit( trX, trY ,
                     epochs=MODEL_EPOCHS,
                     batch_size=MODEL_BATCH_SZ,
                     validation_data=(data['x_val'], data['y_val']))

for i,layer in enumerate(model.layers):
    layer.trainable = False

#add output of model to model_c1
oup_model_data_tr2 = get_oup_model(model, data['partial_x_train_2'])
oup_model_data_val = get_oup_model(model, data['x_val'])
trX                = append_data_col(oup_model_data_tr2, data['partial_x_train_2'])
x_val_c1           = append_data_col(oup_model_data_val, data['x_val'])
trY                = data['partial_y_train_2']
history_causal_1 = model_causal_1.fit( trX, trY,
                                       epochs=MODEL_C1_EPOCHS,
                                       batch_size=MODEL_BATCH_SZ,
                                       validation_data=(x_val_c1, data['y_val']))


for i,layer in enumerate(model_causal_1.layers):
    layer.trainable = False
model_causal_1.summary()

#add output of model to model_c2
oup_model_data_tr3      = get_oup_model(model, data['partial_x_train_3'])
oup_modelc1_data_tr3    = get_oup_model(model_causal_1, append_data_col( oup_model_data_tr3 , data['partial_x_train_3']))
oup_modelc1_data_val    = get_oup_model(model_causal_1, append_data_col( oup_model_data_val , data['x_val']))

trX         = append_2data_col(oup_modelc1_data_tr3, oup_model_data_tr3,  data['partial_x_train_3'])
x_val_c2    = append_2data_col(oup_modelc1_data_val, oup_model_data_val,  data['x_val'])
trY         = data['partial_y_train_3']

history_causal_2= model_causal_2.fit(trX, trY,
                                     epochs=MODEL_C2_EPOCHS,
                                     batch_size=MODEL_BATCH_SZ,
                                     validation_data=(x_val_c2, data['y_val']))

for i,layer in enumerate(model_causal_2.layers):
    layer.trainable = False

x_test_data_comb = data['x_train'][10000:]
y_test_data_comb = data['y_train'][10000:]
oup_model_data_all = get_oup_model_nornd(model, x_test_data_comb)
oup_modelc1_data_all = get_oup_model_nornd(model_causal_1, append_data_col(oup_model_data_all, x_test_data_comb))
oup_modelc2_data_all = get_oup_model_nornd(model_causal_2, append_2data_col(oup_modelc1_data_all, oup_model_data_all, x_test_data_comb))

data_inp_to_comb        = np.zeros([x_test_data_comb.shape[0], 3])
data_inp_to_comb[:,0]   = oup_model_data_all
data_inp_to_comb[:,1]   = oup_modelc1_data_all
data_inp_to_comb[:,2]   = oup_modelc2_data_all

oup_model_data_val  = get_oup_model_nornd(model, data['x_val'])
oup_modelc1_data_val= get_oup_model_nornd(model_causal_1, append_data_col(oup_model_data_val, data['x_val']))
oup_modelc2_data_val= get_oup_model_nornd(model_causal_2, append_2data_col(oup_modelc1_data_val, oup_model_data_val, data['x_val']))

data_val_to_comb      = np.zeros([data['x_val'].shape[0], 3])
data_val_to_comb[:,0] = oup_modelc2_data_val
data_val_to_comb[:,1] = oup_modelc1_data_val
data_val_to_comb[:,2] = oup_model_data_val


history_comb= model_comb.fit(data_inp_to_comb,
                             y_test_data_comb,
                             epochs= MODEL_COMB_EPOCHS,
                             batch_size=16,
                             validation_data=(data_val_to_comb, data['y_val']))

test_data_pred = {}

oup_model_data_test   = get_oup_model_nornd(model, data['x_test'])
oup_modelc1_data_test = get_oup_model_nornd(model_causal_1, append_data_col(oup_model_data_test, data['x_test'] ))
oup_modelc2_data_test = get_oup_model_nornd(model_causal_2, append_2data_col(oup_modelc1_data_test, oup_model_data_test,  data['x_test'] ))

data_val_to_comb = np.zeros([data['x_test'].shape[0], 3])
data_val_to_comb[:,0] = oup_modelc2_data_test
data_val_to_comb[:,1] = oup_modelc1_data_test
data_val_to_comb[:,2] = oup_model_data_test

oup_modelComb_data_test = get_oup_model(model_comb, data_val_to_comb)

test_data_pred['oup_model_data_test']   = oup_model_data_test
test_data_pred['oup_modelc1_data_test'] = oup_modelc1_data_test
test_data_pred['oup_modelc2_data_test'] = oup_modelc2_data_test
test_data_pred['oup_modelComb_data_test'] = oup_modelComb_data_test
test_data_pred['oup_exp'] = data['y_test']

pickle.dump( history.history, open('44history_noncausal.p', 'wb'))
pickle.dump( history_causal_1.history, open('44history_causal1.p', 'wb'))
pickle.dump( history_causal_2.history, open('44history_causal2.p', 'wb'))
pickle.dump( history_comb.history, open('44history_comb.p', 'wb'))
pickle.dump( test_data_pred, open('44pred.p', 'wb'))


