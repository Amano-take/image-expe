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

    def onehot(self, answer):
        onehot = np.zeros((answer.size, 10))
        onehot[np.arange(answer.size), answer] = 1
        onehot = onehot.reshape(self.B, self.C, 1)
        return onehot

    def crossEnans(self, ans):
        ans_onehot = self.onehot(ans)
        self.crossE = (-1/self.B) * np.sum(ans_onehot * np.log(self.y))
        return self.crossE

        
    def anserrate(self, ans):
        output_last = np.reshape(self.y, (self.B, self.C))
        expect = np.argmax(output_last, axis=1)
        return np.count_nonzero(np.equal(ans, expect)) / self.B

    def back(self):
        #最終層を前提->deltaを受け取らない
        delta = ((self.y - self.ans_onehot)/self.B).reshape(self.B, self.C).T
        return delta