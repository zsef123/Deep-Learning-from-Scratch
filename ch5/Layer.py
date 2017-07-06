import numpy as np

class ReLU():
    def __init__(self):
        self.mask = None

    def foward(self, x):        
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx

class Sigmoid():
    def __init__(self):
        self._out = None

    def foward(self, x):
        self._out = 1 / ( 1 + np.exp(-x))
        return self._out

    def backward(self, dout):
        dx = dout * (1.0 - self._out) * self._out
        return dx