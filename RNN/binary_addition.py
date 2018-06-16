import numpy as np
from keras.models import Sequential, load_model
from keras.layers import SimpleRNN, Dense, Input
import matplotlib.pyplot as plt
import pickle
from keras.utils import plot_model




def get_binary_repr(DecArray, MaxLen):
    bin_str_list = [np.binary_repr(X) for X in DecArray]
    bin_arr = np.zeros([np.shape(bin_str_list)[0], MaxLen])
    # binary arr needs to be padded to make all elements of MaxLen
    _tmp = np.zeros([1,MaxLen])
    for  i, bin_str in enumerate(bin_str_list):
        zeros_padded = MaxLen - len(bin_str)
        _tmp[0][: zeros_padded] = np.zeros([1, zeros_padded])
        _tmp[0][zeros_padded: ] = [int(X) for X in bin_str]
        bin_arr[i] = np.flip(_tmp,0)
    return bin_arr

class BinaryAddition():
    def __init__(self, BitLen, NrSamples ):
        self._BitLen = BitLen
        self._NrSamples = NrSamples
        self.test_data = self.generate_test_data()

    def generate_test_data(self):
        min = 0
        max = np.power(2, self._BitLen -2)
        # operands and outputs in decimal
        operand1_dec = np.random.randint(min, max, self._NrSamples)
        operand2_dec = np.random.randint(min, max, self._NrSamples)
        output_dec = operand1_dec + operand2_dec
        #operands in binary
        operand1_bin = get_binary_repr(operand1_dec, self._BitLen)
        operand2_bin = get_binary_repr(operand2_dec, self._BitLen)
        trY = get_binary_repr(output_dec, self._BitLen)
        trX = np.zeros([self._NrSamples, self._BitLen, 2])
        for i in range(0, self._NrSamples):
            for j in range(0, self._BitLen):
                trX[i, j, :] = [operand1_bin[i, j], operand2_bin[i, j]]
        return {'trX': trX, 'trY': trY}

    def train_network(self, NrRNNlayers = 1, NrDenselayers =0, epochs = 100):
        model = Sequential()
        model.add(SimpleRNN(self._BitLen, input_shape=(None, 2), return_sequences=[False, True][NrRNNlayers > 1]))
        for i in range(1,NrRNNlayers):
            model.add(SimpleRNN(self._BitLen, activation='sigmoid'))
        model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
        #history = model.fit(self.test_data['trX'], self.test_data['trY'], epochs=epochs, batch_size=256, validation_split=0.2)
        #pickle.dump(history.history, open("BinAddRnn%dLayersDense%Layer.p", 'wb'))
        #model.save("BinAddRnn%dLayersDense%Laye.h5")
        plot_model(model, to_file='model_plot_1RNN.png', show_shapes=True, show_layer_names=True)
        return {'OupFile': "BinAddRnn%dLayersDense%Layer.p", 'model': "BinAddRnn%dLayersDense%Laye.h5"}


    def plot(self, File):
        binadd = pickle.load(open(File, 'rb'))
        plt.title('Validation and Accuracy')
        plt.subplot(1, 1, 1)
        plt.xlabel('Epochs')
        plt.ylabel('LossAcc')
        epochs = range(1, len(binadd['acc']) + 1)
        plt.plot(epochs, binadd['val_loss'], 'bo', label='comb Val Loss')
        plt.plot(epochs, binadd['val_acc'], 'b+', label='comb Val Acc')
        plt.legend()
        plt.grid()
        plt.show()

    def evaluate_model(data, modelPath, X1, X2, BitLen):
        _model = load_model(modelPath)
        X1 = get_binary_repr(X1, 16)[0]
        X2 = get_binary_repr(X2, 16)[0]
        print(X1.shape)
        trX = np.zeros([BitLen, 2])
        for j in range(0, BitLen):
            trX[j, :] = [X1[j], X2[j]]
        return np.array([round(X[0]) for X in _model.predict(data)])





X = BinaryAddition(16, 20000)

Ret = X.train_network(1,0, 10)
#X.plot(Ret['OupFile'])

#print(X.evaluate_model(Ret['model'], [45], [50], 16))


