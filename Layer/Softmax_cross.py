import numpy as np

class Softmax_cross():

    def __init__(self):
        pass

    def prop(self, x):
        self.x = x
        self.B, self.C, _ = x.shape
        alpha = np.repeat(x.max(axis=1), self.C, axis=1).reshape(self.B, self.C, 1)
        sumexp = np.repeat(
            np.sum(np.exp(x - alpha), axis=1), 10, axis=1).reshape(self.B, self.C, 1)
        self.y = np.exp(x - alpha) / sumexp
        return self.y

    def crossEn(self, ans_onehot):
        self.ans_onehot = ans_onehot
        self.crossE = (-1/self.B) * np.sum(ans_onehot * np.log(self.y))
        return self.crossE

    def back(self):
        #最終層を前提->deltaを受け取らない
        delta = ((self.y - self.ans_onehot)/self.B).reshape(self.B, self.C).T
        return delta