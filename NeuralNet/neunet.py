from neuron import neuron



bias = [1]

##NeuralNetSkeleton = ((1, 7 ), (2, 3), (3, 2))


class neunet():
    def __init__(self, neuralNetSkeleton, inputVecRank, InitialWeight = 0, activationType = "THRES"):
        """ Intializes a neuralnetwork of N layers

        NeuralNet = [ (1,(neuron(), neuron(),..., neuron())),
                      (2,(neuron(), neuron(),..., neuron())),
                       ...
                      (n,(neuron(), neuron(),..., neuron())),


        Args:
            neuralNetSkeleton:  Tupule list of (layer, Number of Neurons)
                                e.g ((1, 7 ), (2, 3), (3, 2)) is a three layer with the first layer having 7 neurons,
                                the second layer of three neurons and the third layer of two neurons

            inputVecRank:       the dimension of the input samples

            InitialWeight:      the initial weights for the neural network

            activationType:     Three kinds of activation functions "THRES", "LINEAR", "SIGMOID"


        """
        self.__neuralNetSkeleton    = neuralNetSkeleton
        self.__neuralNet            = []
        self.__inputVecRank         = inputVecRank

        for layerNumber, noNeuronsInThisLayer  in self.__neuralNetSkeleton :
            weightVectorForThisLayer = [InitialWeight] + [InitialWeight]*inputVecRank
            neuronListForThisLayer   =  tuple([ neuron(weightVector = weightVectorForThisLayer, actFuncType = activationType) for i in range(0, noNeuronsInThisLayer)])
            self.__neuralNet.append( [ layerNumber, neuronListForThisLayer])
            # the input rank or dimension for the next layer is the output dimension of this layer
            # assuming that we have a completely connected feedforward network
            inputVecRank = noNeuronsInThisLayer
        self.__neuralNet.sort()



    def run_iteration(self, inputVec, Expectedoutput = 0):
        """
            The output of the previous layer(outPrevLayer) is the input to the next layer
            and so on. Start from the first layer and compute layer by layer
        """
        assert (len(inputVec) == self.__inputVecRank)
        outPrevLayer = inputVec
        # for the first layer the PrevLayerNumber = 0 i.e the input
        PrevLayerNumber = 0
        for layerNumber, neuronList in self.__neuralNet:
            assert (layerNumber == PrevLayerNumber +1), "Missing a Neural Layer"
            outPrevLayer = [ tmpNeuron.calc_output(bias + outPrevLayer) for j,tmpNeuron in enumerate(neuronList)]
            PrevLayerNumber = layerNumber

        ## Calculate the error


        ## Update the Weights
        print outPrevLayer
        return outPrevLayer




