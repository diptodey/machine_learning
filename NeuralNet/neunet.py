from neuron import neuron


inputVec = [0,1,1,0]

NeuralNet = {"Layer1":
                {"neu1":[neuron(), "1234"],
                 "neu2":[neuron(), "1234"],
                 "neu3":[neuron(), "1234"],
                 "neu4":[neuron(), "1234"]},
             "Layer2":
                {"neu1":[neuron(), "1234"],
                 "neu2":[neuron(), "1234"],
                 "neu3":[neuron(), "1234"],
                 "neu4":[neuron(), "1234"]},
             "Layer3":
                {"neu1":[neuron(), "1234"],
                 "neu2":[neuron(), "1234"],
                 "neu3":[neuron(), "1234"],
                 "neu4":[neuron(), "1234"]}}


def run_iteration():
    for i in range(0,len(NeuralNet)):
        print(" Neuron Layer %d " %i)
        for neu in NeuralNet["Layer"+str(i)]:
            if (i ==1):
                #use the input vector
                neu.  inputVec





run_iteration()