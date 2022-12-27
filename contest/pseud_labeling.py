import mnist
import numpy as np
import sys, os
from decimal import Decimal, ROUND_HALF_UP
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import matplotlib.pyplot as plt
from Layer import BatchNormalize
from Layer import Pooling
from Layer import Conv3D
from Layer import Initialize
from Layer import Softmax_cross
from Layer import Activate
from Layer import Affine
from Layer import Diversify
from Layer.ConAn import ConAn
from Layer import Dropout
from Layer.Imshow import Imshow
from Layer.RandomArgumentation import RArg

class p_labeling():
    def __init__(self):
        self.num = 50
        pass

    def testansrate(self, X, X_test):
        outC = self.Conv.prop(X.reshape(100, 1, 28, -1))
        # プーリング
        outP = self.pooling.pooling(outC, 7)
        # ~中間層
        # 正規化
        outN = self.Bnormal.prop(outP.reshape(outP.shape[0], -1, 1))
        # 中間層
        outS = self.relu.prop(outN)
        # DropOut -> 汎化性能が上がる
        outD = self.dropout.prop(outS)
        # ~最終層
        outAf2 = self.Affine2.prop(outD)
        # 最終層
        finout = self.SofCross.prop(outAf2)
        return self.SofCross.crossEnans(X_test)

    def cnnpractice(self, X, X_test, Y, Y_test, alpha):
        B = 100
        C = 10
        imgl = 28
        poolw = 7

        allX = np.vstack((X, X_test))
        allY = np.vstack((Y, Y_test))
        ini = Initialize.Initialize(allX, allY, B, C, 0)
        epoch = allX.shape[0] // 100
        crossE = 0
        for j in range(epoch):
            percent = Decimal(str(j * 100 / epoch)).quantize(Decimal('0'), rounding=ROUND_HALF_UP)
            print('\r' + "CNN:" + str(percent) + '%', end="")
            Batch_img, onehot, arg = ini.randomselect_with_arg()
            mult = np.repeat(np.where(arg >=  X.shape[0], alpha, 1).reshape(1, -1), 10, axis=0)
            out_Ra = self.ra.prop2(Batch_img, 50)
            outC = self.Conv.prop(out_Ra.reshape(B, 1, imgl, -1))
            # プーリング
            outP = self.pooling.pooling(outC, poolw)
            # ~中間層
            # 正規化
            outN = self.Bnormal.prop(outP.reshape(outP.shape[0], -1, 1))
            # 中間層
            outS = self.relu.prop(outN)
            # DropOut -> 汎化性能が上がる
            outD = self.dropout.prop(outS)
            # ~最終層
            outAf2 = self.Affine2.prop(outD)
            # 最終層
            finout = self.SofCross.prop(outAf2)
            crossE = crossE + self.SofCross.crossEn(onehot)

            delta_outAf2 = self.SofCross.back() * mult
            # print(delta_outAf2.shape) -> C * B
            delta_outD = self.Affine2.back_l2(delta_outAf2)
            # print(delta_outS.shape) -> M * B
            delta_outS = self.dropout.back(delta_outD)
            # print() -> M * B
            delta_outN = self.relu.back(delta_outS)
            # print(delta_outN.shape) -> M * B
            delta_outP = self.Bnormal.back_l2(delta_outN)
            # print(delta_outAf1.shape) -> M * B
            # print(delta_outP.shape) -> (ch * imgsize) * B
            delta_outC = self.pooling.back(delta_outP.T)
            _ = self.Conv.back_l2(delta_outC)
        print("\r" + str(crossE / (j+1)) + "\033[0K")
        

    def renewing(self, Xtest):
        B = 100
        C = 10
        imgl = 28
        poolw = 7
        ans = np.empty((0,C, 1), dtype=float)
        anscheck = np.empty((0,), dtype=int)
        epoch = Xtest.shape[0] // 100
        ini2 = Initialize.Initialize(Xtest, Xtest, 100, 10 , 0)
        for j in range(epoch):
            percent = Decimal(str(j * 100 / epoch)).quantize(Decimal('0'), rounding=ROUND_HALF_UP)
            print('\r' + "renewing:" + str(percent) + '%', end="")
            Batch_img = ini2.orderselect_test(j)
            outC = self.Conv.prop(Batch_img.reshape(B, 1, imgl, -1))
            # プーリング
            outP = self.pooling.pooling(outC, poolw)
            # ~中間層
            # 正規化
            outN = self.Bnormal.prop(outP.reshape(outP.shape[0], -1, 1))
            # 中間層
            outS = self.relu.prop(outN)
            # DropOut -> 汎化性能が上がる
            outD = self.dropout.prop(outS)
            # ~最終層
            outAf2 = self.Affine2.prop(outD)
            # 最終層
            finout = self.SofCross.prop(outAf2)
            finout2 = finout.reshape(B, -1)
            expected = np.argmax(finout2, axis=1)
            ans = np.vstack((ans, finout))
            anscheck = np.hstack((anscheck, expected))
        print("\r" + str(self.ansrate(anscheck[0:200])) + "\033[0K")
        return ans

    def ansrate(self, answer):
        _,  testY = ConAn.get(0, 200)
        return np.count_nonzero(np.equal(answer, testY))

    def study(self):
        B = 100
        C = 10
        imgl = 28
        ch = 32
        filw = 5
        poolw = 7
        l2lambda = 0.0005
        phi = 0.5

        #pseud
        alphaf = 3
        T1 = 10
        T2 = 70

        M = M = ch * (imgl // poolw) * (imgl // poolw)
        X, Y = ConAn.getmnist()
        Y_onehot = ConAn.onehot(Y)
        X_test = ConAn.gettest()
        Y_test_soft = np.random.randint(0, 1, (X_test.shape[0], 10, 1))
        self.ra = RArg(0.1, 10, np.pi/8, 5/4, 5, np.pi/9, 5)
        self.Conv = Conv3D.Conv3D(ch, filw, imgl, B,
                             1, l2lambda, opt="MSGD")
        self.pooling = Pooling.Pooling()
        self.Bnormal = BatchNormalize.BatchNormalize(M, l2lambda, opt="MSGD")
        self.relu = Activate.Relu()
        self.dropout = Dropout.Dropout(phi, B, M)
        self.Affine2 = Affine.Affine(M, C, l2lambda, "MSGD")
        self.SofCross = Softmax_cross.Softmax_cross()

        for i in range(self.num):
            if i < T1:
                alpha = 0
            elif i < T2:
                alpha = (i - T1) * alphaf / (T2 - T1)
            else:
                alpha = alphaf
            print('\r' + str(i+1) + "epoch:\033[0K")
            self.cnnpractice(X, X_test, Y_onehot, Y_test_soft, alpha)
            Y_test = self.renewing(X_test) 
        
        np.savez("./Parameters/pseudo0", self.Affine2.W, self.Affine2.b, self.Bnormal.beta, self.Bnormal.ganma,
                        self.Conv.filter_W, self.Conv.bias)

pl = p_labeling()
pl.study()