from neuron import neuron



bias = [1]
"""
NeuralNet = {"Layer1":{ "neu1":neuron(), "neu2":neuron(),..., "neun":neuron()},
             "Layer2":{ "neu1":neuron(), "neu2":neuron(),..., "neun":neuron()},
             ...
             "LayerN":{ "neu1":neuron(), "neu2":neuron(),..., "neun":neuron()}}

"""
##NeuralNetSkeleton = ((1, 7 ), (2, 3), (3, 2))


class neunet():
    def __init__(self, neuralNetSkeleton, inputVecRank, IniWght = 0, activationType = "THRES"):
        """
            The output of the previous layer is the input to the next layer
            and so on. Start from the first layer and compute layer by layer
        """
        self.__neuralNetSkeleton    = neuralNetSkeleton
        self.__neuralNet            = {}
        outRankPrevLayer            = inputVecRank
        for layer in self.__neuralNetSkeleton :
            tmpLayer = [neuron(weightVector =[IniWght] + [IniWght]*inputVecRank, actFuncType = activationType)]* layer[1]
            self.__neuralNet["Layer" + str(layer[0]) ] = tuple( tmpLayer)
            inputVecRank = layer[1]



    def run_iteration(self, inputVec, Expectedoutput = 0):
        ## Run the Current Iteration based on the Input
        outPrevLayer = inputVec
        for i in range(1,len(self.__neuralNet)+1):
            layer = "Layer" + str(i)
            outCurrLayer = [0]*len(self.__neuralNet[layer])

            for j in range(0,len(self.__neuralNet[layer])):
                outCurrLayer[j-1] = self.__neuralNet[layer][j].calc_output(bias + outPrevLayer)
            print "  Op = " + str(outCurrLayer)
            outPrevLayer = outCurrLayer

        ## Calculate the error


        ## Update the Weights
        print outCurrLayer
        return outCurrLayer




