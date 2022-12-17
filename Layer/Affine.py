import numpy as np
import sys, os
sys.path.append(os.path.dirname(__file__))
import Adam
import MomentumSGD
from Layer.SAM import SAM
#全結合層
class Affine():
    #A -> Bの全結合層
    def __init__(self, A, B, l2para, opt="Adam"):
        #W行列, bバイアス 
        self.l2lambda = l2para
        self.W = np.random.normal(loc=0, scale=np.sqrt(1/A), size=(B, A))
        self.b = np.random.normal(loc=0, scale=np.sqrt(1/A), size=(B, 1))
        #ユニットの数
        self.A = A
        self.B = B
        if(opt == "Adam"):
            #Adamインスタンス化
            self.Wad = Adam.Adam(self.W)
            self.bad = Adam.Adam(self.b)
        elif(opt == "MSGD"):
            self.Wad = MomentumSGD.MSGD(self.W)
            self.bad = MomentumSGD.MSGD(self.b)
        elif(opt == "SAM"):
            self.wsam = SAM(self.W)
            self.bsam = SAM(self.b)
        elif(opt == "ADSGD"):
            self.Wad = MomentumSGD.ADSGD(self.W)
            self.bad = MomentumSGD.ADSGD(self.b)

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

    def SAM_back1(self, dy):
        delta_x = np.matmul(self.W.T, dy)
        #print(delta_x.shape) -> imgsize * B
        delta_W = np.matmul(dy, self.x.reshape(self.Bnum, self.A))
        delta_b = np.sum(dy, axis=1).reshape(self.B, 1)
        self.W = self.wsam.calw(delta_W)
        self.b = self.bsam.calw(delta_b)
        return delta_x
    
    def SAM_back2(self, dy):
        delta_x = np.matmul(self.W.T, dy)
        #print(delta_x.shape) -> imgsize * B
        delta_W = np.matmul(dy, self.x.reshape(self.Bnum, self.A))
        delta_b = np.sum(dy, axis=1).reshape(self.B, 1)
        self.W = self.wsam.update(delta_W)
        self.b = self.bsam.update(delta_b)
        return delta_x

    def update(self, W, b):
        self.W = W
        self.b = b

    def back_ADSGD(self, dy, k):
        delta_x = np.matmul(self.W.T, dy)
        #print(delta_x.shape) -> imgsize * B
        delta_W = np.matmul(dy, self.x.reshape(self.Bnum, self.A))
        delta_b = np.sum(dy, axis=1).reshape(self.B, 1)
        self.W = self.Wad.update(delta_W, k)
        self.b = self.bad.update(delta_b, k)
        return delta_x
            
    def back_l2(self, dy):
        delta_x = np.matmul(self.W.T, dy)
        #print(delta_x.shape) -> imgsize * B
        delta_W = np.matmul(dy, self.x.reshape(self.Bnum, self.A)) + self.l2lambda * 2 * self.W
        delta_b = np.sum(dy, axis=1).reshape(self.B, 1) + self.l2lambda * 2 * self.W
        self.W = self.Wad.update(delta_W)
        self.b = self.bad.update(delta_b)
        return delta_x


    
