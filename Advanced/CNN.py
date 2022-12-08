import mnist
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from Layer import BatchNormalize
from Layer import Pooling
from Layer import Conv3D
from Layer import Initialize
from Layer import Softmax_cross
from Layer import Activate
from Layer import Affine
import numpy as np


# 一番シンプルなもの


class test():

    def __init__(self, a):
        seed = 600
        np.random.seed(seed)
        self.B = 100
        self.M = 300
        self.C = 10
        self.poolw = 2
        self.ch = 16
        self.filw = 5

        if a == 0:
            self.study()
        else:
            self.test()

    def study(self):
        # 層に関して
        B = self.B
        M = self.M
        C = self.C
        # 画像データ
        X = mnist.download_and_parse_mnist_file("train-images-idx3-ubyte.gz")
        Y = mnist.download_and_parse_mnist_file("train-labels-idx1-ubyte.gz")
        Ini = Initialize.Initialize(X, Y, B, C)
        # 過学習制御用
        Xtest = mnist.download_and_parse_mnist_file(
            "t10k-images-idx3-ubyte.gz")
        Ytest = mnist.download_and_parse_mnist_file(
            "t10k-labels-idx1-ubyte.gz")
        imgl = X.shape[1]
        # 学習に関して
        num = 20
        epoch = X.shape[0] // B
        # crossE初期化
        crossE = 0
        correctnum = 0
        # Layer
        Conv = Conv3D.Conv3D(self.ch, self.filw, imgl, B, 1)
        pooling = Pooling.Pooling()
        Affine1 = Affine.Affine(
            self.ch * (imgl // self.poolw) * (imgl // self.poolw), M)
        Bnormal = BatchNormalize.BatchNormalize(M)
        Sigmoid = Activate.Sigmoid()
        Affine2 = Affine.Affine(M, C)
        SofCross = Softmax_cross.Softmax_cross()
        for i in range(num):
            precrossE = crossE
            crossE = 0
            for j in range(epoch):
                Batch_img, onehot = Ini.randomselect()
                Batch_img = Batch_img.reshape(self.B, 1, imgl, -1)
                # 画像畳み込み
                outC = Conv.prop(Batch_img)
                # プーリング
                outP = pooling.pooling(outC, self.poolw)
                # 形変える
                outP = outP.reshape(outP.shape[0], -1, 1)
                # ~中間層
                outAf1 = Affine1.prop(outP)
                # 正規化
                outN = Bnormal.prop(outAf1)
                # 中間層
                outS = Sigmoid.prop(outN)
                # ~最終層
                outAf2 = Affine2.prop(outS)
                # 最終層
                finout = SofCross.prop(outAf2)
                crossE = crossE + SofCross.crossEn(onehot)

                # 学習
                delta_outAf2 = SofCross.back()
                # print(delta_outAf2.shape) -> C * M
                delta_outS = Affine2.back(delta_outAf2)
                # print(delta_outS.shape) -> M * B
                delta_outN = Sigmoid.back(delta_outS)
                # print(delta_outN.shape) -> M * B
                delta_outAf1 = Bnormal.back(delta_outN)
                # print(delta_outAf1.shape) -> M * B
                delta_outP = Affine1.back(delta_outAf1)
                # print(delta_outP.shape) -> (ch * imgsize) * B
                delta_outC = pooling.back(delta_outP.T)
                _ = Conv.back(delta_outC)

            #1epoch終了

            crossE = crossE / epoch
            print(crossE)
            if(crossE < precrossE):
                np.savez("./Parameters/test", Affine1.W, Affine1.b, Affine2.W, Affine2.b, Bnormal.beta, Bnormal.ganma,
                         Conv.filter_W, Conv.bias)
            precorrect = correctnum 
            correctnum = 0
            #過学習判定
            test = Initialize.Initialize(Xtest, Ytest, self.B, self.C)
            #テストデータB * 10仕様
            for i in range(10):
                Batch_img, ans = test.orderselect(i)
                Batch_img = Batch_img.reshape(self.B, 1, imgl, -1)
                # 画像畳み込み
                outC = Conv.prop(Batch_img)
                # プーリング
                outP = pooling.pooling(outC, self.poolw)
                # 形変える
                outP = outP.reshape(outP.shape[0], -1, 1)
                # ~中間層
                outAf1 = Affine1.prop(outP)
                # 正規化
                outN = Bnormal.prop(outAf1)
                # 中間層
                outS = Sigmoid.prop(outN)
                # ~最終層
                outAf2 = Affine2.prop(outS)
                # 最終層
                finout = SofCross.prop(outAf2)
                crossE = crossE + SofCross.crossEn(onehot)
                output_last = np.reshape(finout, (self.B, self.C))
                expect = np.argmax(output_last, axis=1)
                for i, an in enumerate(ans):
                    if an == expect[i]:
                        correctnum = correctnum + 1
            print(correctnum)
            if(correctnum < precorrect - 10): 
                print("overfitting!")
                break

        print("finish!")

    def test(self):
        # 層に関して
        B = self.B
        M = self.M
        C = self.C
        parameters = np.load("./Parameters/test.npz")
        W1 = parameters['arr_0']
        W2 = parameters['arr_2']
        b1 = parameters['arr_1']
        b2 = parameters['arr_3']
        normal_beta = parameters['arr_4']
        noraml_ganma = parameters['arr_5']
        filter_W = parameters['arr_6']
        fil_bias = parameters['arr_7']

        Xtest = mnist.download_and_parse_mnist_file(
            "t10k-images-idx3-ubyte.gz")
        Ytest = mnist.download_and_parse_mnist_file(
            "t10k-labels-idx1-ubyte.gz")
        Ini = Initialize.Initialize(Xtest, Ytest, self.B, self.C)
        imgl = Xtest.shape[1]

        epoch = Xtest.shape[0] // self.B
        img_size = Xtest[0].size
        correct_num = 0
        Conv = Conv2D.conv2D(self.ch, self.filw, Xtest.shape[1], B)
        pooling = Pooling.Pooling()
        Affine1 = Affine.Affine(
            self.ch * (imgl // self.poolw) * (imgl // self.poolw), M)
        Bnormal = BatchNormalize.BatchNormalize(M)
        Sigmoid = Activate.Sigmoid()
        Affine2 = Affine.Affine(M, C)
        SofCross = Softmax_cross.Softmax_cross()
        for i in range(epoch):
            Batch_img, ans = Ini.orderselect(i)
            # 画像畳み込み
            outC = Conv.test(Batch_img, filter_W, fil_bias)
            # プーリング
            outP = pooling.pooling(outC, self.poolw)
            # ~中間層
            outAf1 = Affine1.test(W1, b1, outP)
            # 正規化
            outN = Bnormal.test(outAf1, normal_beta, noraml_ganma)
            # 中間層
            outS = Sigmoid.prop(outN)
            # ~最終層
            outAf2 = Affine2.test(W2, b2, outS)
            # 最終層
            finout = SofCross.prop(outAf2)

            output_last = np.reshape(finout, (self.B, self.C))
            expect = np.argmax(output_last, axis=1)
            for i, an in enumerate(ans):
                if an == expect[i]:
                    correct_num = correct_num + 1
        print(correct_num * 100 / Xtest.shape[0])


print("study -> 0, test -> 1")
a = int(input())
test(a)
