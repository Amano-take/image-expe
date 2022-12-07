import numpy as np

class Initialize():
    def __init__(self, X, Y, B, C):
        #Bはtestを割り切る数を想定
        #->convがうまくいかない
        self.X = X
        self.Y = Y
        self.B = B
        self.C = C

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
        # 正解をone-hot vectorに
        onehot = self.onehot(answer)
        return Batch_img, onehot

    def orderselect(self, i):
        Batch_img = self.X[i*self.B : (i+1)*self.B]
        #answer = self.Y[i*self.B : (i+1)*self.B]
        return Batch_img #, answer