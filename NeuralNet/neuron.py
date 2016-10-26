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

from activationfuns import *

# Basic neuron class , the name is a misnomer TODO
import numpy as np

class neuron():
    def __init__(self, weightVector = [1,2,3,4], actFuncType = "THRES", sigmoid_slope = 1):
        print("                 neuron created")
        ## TODO  How to incorporate sigmoid slope in a nice way
        ## TODO  Incorporate User Defined Activation Functions
        self.__actFunc = {"THRES": activation_threshold, "LINEAR": activation_linear, "SIGMOID": activation_sigmoid}[actFuncType]
        self._weightVector = weightVector
        self.__sigmoid_slope = 1
        print("Created neuron")


    def update_weights(self,newWeight):
        self._weightVector = newWeightVector


    def calc_output(self,inputVector):
        # input0 is for bias is alaways 1
        inputVector+=[1]
        print inputVector ,self._weightVector
        return self.__actFunc(np.dot( self._weightVector,  np.transpose(inputVector)))


    def get_weights(self):
        return self._weightVector





