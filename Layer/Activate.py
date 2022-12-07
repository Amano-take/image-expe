import numpy as np

class Sigmoid():
    sigmoid_range = 34.53877639410684
    def __init__(self):
        pass
    @staticmethod
    @np.vectorize
    def sigmoid(x):
        if x <= - Sigmoid.sigmoid_range:
            return 1e-15
        if x >= Sigmoid.sigmoid_range:
            return 1.0 - 1e-15
        return 1.0 / (1.0 + np.exp(-x))

    def prop(self, x):
        self.y = Sigmoid.sigmoid(x)
        self.B, self.M, _ = x.shape
        return self.y

    def back(self, delta):
        return delta *  (1 - self.y.reshape(self.B, self.M).T) * self.y.reshape(self.B, self.M).T
