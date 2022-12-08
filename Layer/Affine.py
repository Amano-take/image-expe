import numpy as np
import sys, os
sys.path.append(os.path.dirname(__file__))
import Adam
#全結合層
class Affine():
    #A -> Bの全結合層
    def __init__(self, A, B):
        #W行列, bバイアス 
        self.W = np.random.normal(loc=0, scale=np.sqrt(1/A), size=(B, A))
        self.b = np.random.normal(loc=0, scale=np.sqrt(1/A), size=(B, 1))
        #ユニットの数
        self.A = A
        self.B = B
        #Adamインスタンス化
        self.Wad = Adam.Adam(self.W)
        self.bad = Adam.Adam(self.b)
    
    def test(self, W, b, x):
        return np.matmul(W, x) + b

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

    
