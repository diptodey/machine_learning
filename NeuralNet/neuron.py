#-------------------------------------------------------------------------------
# Name:        NeuralNet
# Purpose:
#
# Author:      Dipto
#
# Created:     25-10-2016
# Copyright:   (c) Dipto 2016
# Licence:     <your licence>
#-------------------------------------------------------------------------------


# Basic neuron class , the name is a misnomer TODO
import numpy as np

class neuron():
    def __init__(self, weightVector = [1,2,3,4], actFuncType = "THRES", sigmoid_slope = 1):
        self.__actFuncType =  actFuncType
        self.__weightVector = weightVector
        self.__sigmoid_slope = 1


    def update_weights(self,newWeight):
        self.__weightVector = newWeightVector


    def calc_output(self,inputVector):
        v=np.dot( self.__weightVector,  np.transpose(inputVector))
        if self.__actFuncType == "THRES":
            return self.activation_threshold(v)
        elif self.__actFuncType == "LINEAR":
            return self.activation_linear(v)
        elif self.__actFuncType == "SIGMOID":
            return self.activation_sigmoid(v)
        else:
            assert 1, (" Undefined activation function type")


    def get_weights(self):
        return self.__weightVector


    def activation_threshold(self,v):
        if v < 0:
            return 0
        else:
            return 1


    def activation_linear(self,v):
        if v >= 0.5:
            return 1
        elif v > -0.5:
            return v
        else:
            return -1


    def activation_sigmoid(self,v):
        return 1/(1 + np.exp(-self.__sigmoid_slope*v))


    def activation_stochastic(self,v):
        pass




