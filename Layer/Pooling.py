import numpy as np


class Pooling():

    def __init__(self):
        pass

    def pooling(self, x, w):
        # w*wから最大値を取得する
        # x=B* ch *iml*imwとする
        self.w = w
        self.B, self.ch, self.iml, self.imw = x.shape
        self.outl = self.iml // w
        self.outw = self.imw // w
        X = self.x2X(x)
        self.arg_max = np.argmax(X, axis = 1)
        output1 = np.max(X, axis=1).reshape(
            self.B, self.ch, self.outl, self.outw)
        output2 = output1.transpose(1, 0, 2, 3).reshape(self.B, -1, 1)
        return output1

    def x2X(self, x):
        w = self.w
        bigL = self.iml - w + 1
        bigW = self.imw - w + 1
        arrayX = np.zeros((self.B, self.ch, w, w, self.outl, self.outw))
        for i in range(w):
            for j in range(w):
                arrayX[:, :, i, j, :, :] = x[:, :, i:i+bigL:w, j:j+bigW:w]
        X = arrayX.transpose(0, 1, 4, 5, 2, 3).reshape(-1, w*w)
        return X

    def back(self, delta):
        #delta -> B * (ch* imglen)
        w = self.w
        delta_x = np.zeros((delta.size, w * w))
        delta_x[np.arange(self.arg_max.size), self.arg_max.flatten()] = delta.flatten()
        return self.X2x(delta_x)# -> B * ch * imglen * imlen

    def X2x(self, X):
        w = self.w
        bigL = self.iml - w + 1
        bigW = self.imw - w + 1
        x = np.zeros((self.B, self.ch, self.iml, self.imw))
        arX = X.reshape(self.B, self.ch, self.outl, self.outw, w, w).transpose(0, 1, 4, 5, 2, 3)
        for i in range(w):
            for j in range(w):
                x[:, :, i:i+bigL:w, j:j+bigW:w] = arX[:, :, i, j, :, :]
        return x


"""
poo = Pooling()

print(poo.pooling(np.arange(96).reshape(2, 3, 4, 4), 2))
delta = np.arange(24).reshape(3 *4, 2)
print(delta)
print(poo.back(delta))
"""
