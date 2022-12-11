import numpy as np
import matplotlib.pyplot as plt
from pylab import cm

class Initialize():
    def __init__(self, X, Y, B, C, p):
        #Bはtestを割り切る数を想定
        #->convがうまくいかない
        self.X = X
        self.Y = Y
        self.B = B
        self.C = C
        self.p = p

    def onehot(self, answer):
        onehot = np.zeros((answer.size, 10))
        onehot[np.arange(answer.size), answer] = 1
        onehot = onehot.reshape(self.B, self.C, 1)
        return onehot

    def randomselect(self):
        batch_random = np.random.randint(0, self.X.shape[0], self.B)
        # 画像取得
        Batch_img = np.array(self.X[batch_random])
        # 正解を取得
        answer = np.array(self.Y[batch_random])
        #plt.imshow(Batch_img[0], cmap=cm.gray)
        #plt.show()
        #print(answer[0])
        # 正解をone-hot vectorに
        onehot = self.onehot(answer)
        return Batch_img, onehot, answer

    def orderselect(self, i):
        Batch_img = self.X[i*self.B : (i+1)*self.B]
        answer = self.Y[i*self.B : (i+1)*self.B]
        return Batch_img, answer

    def orderselect_test(self, i):
        Batch_img = self.X[i*self.B : (i+1)*self.B]
        return Batch_img

    def labelSmoothing(self):
        batch_random = np.random.randint(0, self.X.shape[0], self.B)
        # 画像取得
        Batch_img = np.array(self.X[batch_random])
        # 正解を取得
        answer = np.array(self.Y[batch_random])
        #plt.imshow(Batch_img[0], cmap=cm.gray)
        #plt.show()
        #print(answer[0])
        # 正解をsmoothing vector に
        smooth = self.smooth(answer, self.p)
        return Batch_img, smooth, answer

    def smooth(self, answer, eps):
        onehot = np.zeros((answer.size, 10))
        smooth = onehot + eps/9
        smooth[np.arange(answer.size), answer] = 1 - eps
        smooth = smooth.reshape(self.B, self.C, 1)
        return smooth