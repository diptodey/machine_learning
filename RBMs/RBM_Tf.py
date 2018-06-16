

import pickle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class DBM():
    def __init__(self, layers, lrate, epochs, batch_size, train_data, val_data,
                 stop_at, dump_weight_path):
        self.weight_list = []
        self.train_data = train_data
        self.val_data = val_data
        self.stop_at = stop_at
        self.dump_weight_path = dump_weight_path
        self.batch_size = batch_size
        self.layers = layers
        self.lrate = lrate
        self.epochs = epochs
        self.energy_diff = []

    def run(self, cd):
        next_train = self.train_data
        next_val = self.val_data
        self.energy_diff = []
        self.weight_list = []
        for count, dim in enumerate(self.layers):
            r1 = RBM(NrV=dim[0], NrH=dim[1],
                     lrate=self.lrate,
                     cd=cd,
                     savepath="./_tmp/model/RBM" + str(count) + ".chp", stop_at=self.stop_at[count])
            r1.run_cd1_minibatch(next_train, next_val, self.epochs, self.batch_size)
            self.energy_diff.append(r1.get_energy_diff())
            self.weight_list.append(r1.get_weights())
            next_train = r1.run_visible(next_train )
            next_val = r1.run_visible(next_val)
        pickle.dump(self.energy_diff, open(self.dump_weight_path + 'energy_diff_cd' + str(cd), "wb"))
        pickle.dump(self.weight_list, open(self.dump_weight_path + 'weights_cd' + str(cd), "wb"))

    def get_min_energy_step(self, layer, cd):
        tmp_energy_gap = pickle.load(open(self.dump_weight_path + 'energy_diff_cd' + str(cd), 'rb'))
        return np.argmin(tmp_energy_gap[layer]) + 1

    def plot_energy_diff(self, layer, cd1=True, cd5=True, cd10=True):
        plt.title('Energy Difference Test and Validation')
        plt.subplot(1, 1, 1)
        plt.xlabel('Epochs')
        plt.ylabel('Energy Difference')
        if cd1:
            cd1_energy_diff = pickle.load(open(self.dump_weight_path + 'energy_diff_cd1', 'rb'))
            epochs = range(1, np.shape(cd1_energy_diff[layer])[0] + 1)
            plt.plot(epochs, cd1_energy_diff[layer], 'b-', label='CD1')
        if cd5:
            cd5_energy_diff = pickle.load(open(self.dump_weight_path + 'energy_diff_cd5', 'rb'))
            epochs = range(1, np.shape(cd5_energy_diff[layer])[0] + 1)
            plt.plot(epochs, cd5_energy_diff[layer], 'r-', label='CD5')
        if cd10:
            cd10_energy_diff = pickle.load(open(self.dump_weight_path + 'energy_diff_cd10', 'rb'))
            epochs = range(1, np.shape(cd10_energy_diff[layer])[0] + 1)
            plt.plot(epochs, cd10_energy_diff[layer], 'y-', label='CD10')
        plt.legend()
        plt.grid()
        plt.savefig(self.dump_weight_path + "EnergyVsCd_layer_" + str(layer) + ".png")
        plt.show()


class RBM():
    def __init__(self, NrV, NrH, lrate=0.01, savepath = "./_tmp/model/RBM1.chp", stop_at = 100, cd = 1):
        self.NrV = NrV
        self.NrH = NrH
        self.savepath = savepath
        self.stop_at = stop_at
        self.curr_epoch_count = 0
        network_weights = self._initialize_weights()
        self.weights = network_weights
        self.W = tf.placeholder(tf.float32, [NrV, NrH])
        self.V_W_bias = tf.placeholder(tf.float32, [self.NrV])
        self.H_W_bias = tf.placeholder(tf.float32, [self.NrH])
        self.data = tf.placeholder(tf.float32,[None, self.NrV])

        self.n_w = np.zeros([self.NrV, self.NrH], np.float32)
        self.n_vb = np.zeros([self.NrV], np.float32)
        self.n_hb = np.zeros([self.NrH], np.float32)
        self.o_w = np.random.normal(0.0, 0.01, [self.NrV, self.NrH])
        self.o_vb = np.zeros([self.NrV], np.float32)
        self.o_hb = np.zeros([self.NrH], np.float32)
        self.test_energy = []
        self.val_energy = []

        self.pos_hidden_prob = tf.nn.sigmoid(tf.matmul(self.data, self.W) + self.H_W_bias)
        self.pos_hidden_state = tf.nn.relu(tf.sign(self.pos_hidden_prob -
                                                   tf.random_uniform(tf.shape(self.pos_hidden_prob))))

        recons_hidden_state = self.pos_hidden_state
        for i in range(0, cd):
            neg_visible_prob = tf.nn.sigmoid(tf.matmul(recons_hidden_state, tf.transpose(self.W)) + self.V_W_bias)
            neg_hidden_prob = tf.nn.sigmoid(tf.matmul(neg_visible_prob, self.W) + self.H_W_bias)

        pos_associations = tf.matmul(tf.transpose(self.data), self.pos_hidden_prob)
        neg_associations = tf.matmul(tf.transpose(neg_visible_prob), neg_hidden_prob)

        self.update_w = self.W + (lrate * ((pos_associations - neg_associations) / tf.to_float(tf.shape(self.data)[0])))
        self.update_vb = self.V_W_bias + (lrate * tf.reduce_mean(self.data - neg_visible_prob, 0))
        self.update_hb = self.H_W_bias + (lrate * tf.reduce_mean(self.pos_hidden_prob - neg_hidden_prob, 0))
        self.error = tf.reduce_sum(tf.square(self.data - neg_visible_prob))/tf.to_float(tf.shape(self.data)[1])

        v_bias = tf.transpose(tf.reshape(self.V_W_bias, [1, self.NrV]))
        h_bias = tf.transpose(tf.reshape(self.H_W_bias, [1, self.NrH]))
        self.energy = (-tf.reduce_sum(tf.multiply(tf.matmul(self.data, self.W), self.pos_hidden_state)) -\
                       tf.reduce_sum(tf.matmul(self.data, v_bias)) - \
                       tf.reduce_sum(tf.matmul(self.pos_hidden_state, h_bias)))/tf.to_float(tf.shape(self.data)[0])

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        self.save_weights(self.savepath)

    def _initialize_weights(self):
        # These weights are only for storing and loading model for tensorflow Saver.
        all_weights = dict()
        all_weights['W'] = tf.Variable(tf.random_normal([self.NrV, self.NrH], stddev=0.01, dtype=tf.float32),name='W')
        all_weights['V_W_bias'] = tf.Variable(tf.zeros([self.NrV], dtype=tf.float32), name='V_W_bias')
        all_weights['H_W_bias'] = tf.Variable(tf.random_uniform([self.NrH], dtype=tf.float32), name='H_W_bias')
        return all_weights

    def restore_weights(self, path):
        saver = tf.train.Saver({'W': self.weights['W'],
                                'V_W_bias': self.weights['V_W_bias'],
                                'H_W_bias': self.weights['H_W_bias']})
        saver.restore(self.sess, path)
        self.o_w = self.weights['W'].eval(self.sess)
        self.o_vb = self.weights['V_W_bias'].eval(self.sess)
        self.o_hb = self.weights['H_W_bias'].eval(self.sess)

    def save_weights(self, path):
        self.sess.run(self.weights['W'].assign(self.o_w))
        self.sess.run(self.weights['V_W_bias'].assign(self.o_vb))
        self.sess.run(self.weights['H_W_bias'].assign(self.o_hb))
        saver = tf.train.Saver({'W': self.weights['W'],
                                'V_W_bias': self.weights['V_W_bias'],
                                'H_W_bias': self.weights['H_W_bias']})
        saver.save(self.sess, path, write_meta_graph=False)

    def load_weights(self, w):
        self.o_w = w

    def get_weights(self):
        return {'W': self.o_w, 'V_W_bias': self.o_vb, 'H_W_bias': self.o_hb}

    def run_cd1(self, train_data, epochs):
        for epoch in range(0, epochs):
            self.curr_epoch_count = self.curr_epoch_count +1
            if self.curr_epoch_count == self.stop_at:
                break;
            self.n_w, self.n_vb, self.n_hb = self.sess.run([self.update_w, self.update_vb, self.update_hb],
                                                           feed_dict={self.data: train_data, self.W: self.o_w,
                                                                      self.V_W_bias: self.o_vb, self.H_W_bias: self.o_hb})
            self.o_w = self.n_w
            self.o_vb = self.n_vb
            self.o_hb = self.n_hb
            print(self.get_error(train_data))

    def run_cd1_minibatch(self, train_data, val_data, epochs, batchsz = 100):
        self.test_energy = []
        self.val_energy = []
        for i in range(0,np.int(np.size(train_data,0)/batchsz) ):
            self.run_cd1(train_data[batchsz*i: batchsz*(i+1)], epochs)
            self.sess.run(self.energy, feed_dict={self.data: train_data, self.W: self.o_w, self.V_W_bias: self.o_vb, self.H_W_bias: self.o_hb})
            self.test_energy.append(self.get_energy(train_data))
            self.val_energy.append(self.get_energy(val_data))
            if self.curr_epoch_count == self.stop_at:
                break;
        self.save_weights(self.savepath)

    def plot_energy(self, filename):
        epochs = range(1, len(self.test_energy) + 1)
        plt.title('Energy Test and Val')
        plt.subplot(1, 1, 1)
        plt.xlabel('Epochs')
        plt.ylabel('Energy')
        plt.plot(epochs, np.subtract( self.val_energy, self.test_energy), 'b-', label='Validation and Test Energy Gap')
        plt.legend()
        plt.grid()
        plt.savefig(filename)
        plt.show()

    def get_energy_diff(self):
        return np.subtract(self.val_energy, self.test_energy)

    def run_visible(self, data):
        self.restore_weights(self.savepath)
        feed_dict = {self.data: data, self.W: self.o_w, self.V_W_bias: self.o_vb, self.H_W_bias: self.o_hb}
        hidden_state = self.sess.run(self.pos_hidden_state,  feed_dict=feed_dict)
        return hidden_state

    def get_energy(self, data):
        feed_dict = {self.data: data, self.W: self.o_w, self.V_W_bias: self.o_vb, self.H_W_bias: self.o_hb}
        energy = self.sess.run(self.energy, feed_dict=feed_dict)
        return energy

    def get_error(self, data):
        feed_dict = {self.data: data, self.W: self.o_w,self.V_W_bias: self.o_vb, self.H_W_bias: self.o_hb}
        error = self.sess.run(self.error, feed_dict= feed_dict)
        return error

    def run_hidden(self, data):
        pass


