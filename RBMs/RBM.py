

import numpy as np
import pickle, time

Logistic = lambda x: 1.0 / (1 + np.exp(-x))

class RBM():

    def __init__(self, NrV, NrH):
        """
         Initialize the weight matrix
         NrV: Number of Visible Units
         NrH: Number of Hidden Units


        """
        self.NrV = NrV
        self.NrH = NrH
        np_rng = np.random.RandomState(1234)
        W = np.asarray(np_rng.uniform(low=-0.1 * np.sqrt(6. / (NrV + NrH)),
                                           high=0.1 * np.sqrt(6. / (NrV + NrH)),
                                           size=(NrV, NrH)))
        # Insert weights for the bias units into the first row and first column.
        # the weight matrix is noe a combined for biases accounted
        W = np.insert(W, 0, 0, axis=0)
        W = np.insert(W, 0, 0, axis=1)
        self.W = W


    def train(self, data_train, epochs = 100, lrate = 0.1):
        # add ones for the bias inputs
        data_train = np.insert(data_train, 0, 1, axis=1)
        # data_train is 2d array, each row corresponds to a trianing case
        # we do batch training, number of col
        NrTrainingCases, TrainingCaseDim = np.shape(data_train)

        print(TrainingCaseDim, NrTrainingCases)

        for epoch in range(0,epochs):
            # Clamp to the data and sample from the hidden units.
            # (This is the "positive CD phase", aka the reality phase.)

            pos_hidden_act = np.dot(data_train, self.W)
            pos_hidden_prob = Logistic(pos_hidden_act)
            pos_hidden_prob[:, 0] = 1  # Fix the bias unit.
            pos_associations = np.dot(data_train.T, pos_hidden_prob )
            """
            It is very important to make these hidden states binary, rather than using the probabilities
            themselves. If the probabilities are used, each hidden unit can communicate a real-value to the
            visible units during the reconstruction. This seriously violates the information bottleneck created by
            the fact that a hidden unit can convey at most one bit (on average). This information bottleneck
            acts as a strong regularizer.
            """
            pos_hidden_state = pos_hidden_prob > np.random.rand(NrTrainingCases, self.NrH +1)
            #print(pos_hidden_state)


            # Reconstruct the visible units and sample again from the hidden units.
            # (This is the "negative CD phase", aka the daydreaming phase.)
            neg_visible_act = np.dot(pos_hidden_state, self.W.T)
            neg_visible_prob = Logistic(neg_visible_act)
            neg_visible_prob[:, 0] = 1  # Fix the bias unit.

            neg_hidden_act = np.dot(neg_visible_prob, self.W)
            neg_hidden_prob = Logistic(neg_hidden_act)
            neg_associations = np.dot(neg_visible_prob.T, neg_hidden_prob)

            #print(np.shape(neg_associations))

            # Update weights.
            self.W += lrate * ((pos_associations - neg_associations) / NrTrainingCases)
            error = np.sum((data_train - neg_visible_prob) ** 2)

            print(error)




start_time = time.time()
r = RBM(NrV = 6, NrH = 4)
training_data = np.array([[1,1,1,0,0,0],[1,0,1,0,0,0],[1,1,1,0,0,0],[0,0,1,1,1,0], [0,0,1,1,0,0],[0,0,1,1,1,0]])
#training_data = np.array(data['x_p_train'])
r.train(training_data, epochs = 1000)
elapsed_time = time.time() - start_time
print("elapsed time = %d", elapsed_time)