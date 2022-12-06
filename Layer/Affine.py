import numpy as np
import Adam

#全結合層
class Affine():
    #A -> Bの全結合層
    def __init__(self, W, b):
        #W行列, bバイアス
        self.W = W
        self.b = b
        #ユニットの数
        self.A = W.shape[1]
        self.B = W.shape[0]
        #Adamインスタンス化
        self.Wad = Adam.Adam(W)
        self.bad = Adam.Adam(b)


    def prop(self, x):
        self.x = x
        self.Bnum = x.shape[0]
        return np.matmul(self.W, x) + self.b

    def back(self, dy):
        delta_x = np.matmul(self.W.T, dy)
        #print(delta_x.shape) -> imgsize * B
        delta_W = np.matmul(dy, self.x.reshape(self.Bnum, self.A))
        delta_b = np.sum(dy, axis=1).reshape(self.B, 1)
        self.W = self.Wad.update(delta_W)
        self.b = self.bad.update(delta_b)
        return delta_x

    
