import numpy as np

def activation_threshold(v):
    if v < 0:
         return 0
    else:
         return 1

def activation_linear(v):
    if v >= 0.5:
        return 1
    elif v > -0.5:
        return v
    else:
        return -1

def activation_sigmoid(v):
    return 1/(1 + np.exp(-1*v))


def activation_stochastic(v):
    pass