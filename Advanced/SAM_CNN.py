import mnist
import sys
import os
import matplotlib.pyplot as plt
from pylab import cm
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
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
from Layer import Normalize
import numpy as np

#todo
#Adamからの変更
#足りてないアーギュメンテーションは？
#アーギュメンテーションのハイパラ？
#苦手な問題を繰り返し学習

class test():

    def __init__(self, a):
        seed = 600
        np.random.seed(seed)
        self.B = 100
        self.C = 10
        #4だと学習が早くかなり汎化性能ある。2のメリット見つからず
        #さらに大きく7!!!
        self.poolw = 7
        #8, 16, 4, 32　をためしてみる ->それほど変わらず,,
        #ちょうどいいのは16
        self.ch = 16
        #小さいほうがいいらしい？？
        self.filw = 5
        self.phi = 0.5
        #逆効果
        self.smoothp = 0

        if a == 0:
            self.study()
        else:
            self.test()

    def study(self):
        # 層に関して
        X = np.array(mnist.download_and_parse_mnist_file("train-images-idx3-ubyte.gz"))
        Y = np.array(mnist.download_and_parse_mnist_file("train-labels-idx1-ubyte.gz"))
        imgl = X.shape[1]
        B = self.B
        M = self.ch * (imgl // self.poolw) * (imgl // self.poolw)
        C = self.C
        Ini = Initialize.Initialize(X, Y, B, C, self.smoothp)
        # 過学習制御用　-> 始め200個を使ってることになりますね(´;ω;｀)
        Xtest, Ytest = ConAn.get(0, 200)
        
        # 学習に関して -> 過学習を検知して自動で止まるので大きく
        # 実際に優秀なパラを得たのは手動のearly-stoppingでです
        num = 1000
        epoch = X.shape[0]// B
        # 初期化
        crossE = 0
        correctnum = 0
        precorrectnum = 0
        # Layer
        #ここのハイパラは見てわかる程度に制限しています.
        ra = RArg(0.01, 14, np.pi/6, 6/5, 7, np.pi/18, 5)
        Conv = Conv3D.Conv3D(self.ch, self.filw, imgl, B, 1, opt="SAM")
        pooling = Pooling.Pooling()
        Bnormal = BatchNormalize.BatchNormalize(M, opt="SAM")
        relu = Activate.Relu()
        dropout = Dropout.Dropout(self.phi, B, M)
        Affine2 = Affine.Affine(M, C, "SAM")
        SofCross = Softmax_cross.Softmax_cross()
        #途中から学習
        """
        parameters = np.load("./Parameters/CNN-4.npz")
        W2 = parameters['arr_0']
        b2 = parameters['arr_1']
        normal_beta = parameters['arr_2']
        noraml_ganma = parameters['arr_3']
        filter_W = parameters['arr_4']
        fil_bias = parameters['arr_5']
        Affine2.update(W2, b2)
        Bnormal.update(normal_beta, noraml_ganma)
        Conv.update(filter_W, fil_bias)"""
        for i in range(num):
            precrossE = crossE
            crossE = 0
            for j in range(epoch):
                Batch_img, onehot, ans = Ini.randomselect()
                #argumentation
                Batch_img = ra.prop(Batch_img)
                Batch_img = Batch_img.reshape(self.B, 1, imgl, -1)
                # 画像畳み込み
                outC = Conv.prop(Batch_img)
                # プーリング
                outP = pooling.pooling(outC, self.poolw)
                # 形変える
                outP = outP.reshape(outP.shape[0], -1, 1)
                # ~中間層
                # 正規化
                outN = Bnormal.prop(outP)
                # 中間層
                outS = relu.prop(outN)
                # DropOut -> 汎化性能が上がる
                outD = dropout.prop(outS)
                # ~最終層
                outAf2 = Affine2.prop(outD)
                # 最終層
                finout = SofCross.prop(outAf2)
                crossE = crossE + SofCross.crossEn(onehot)
                """if (j % (epoch // 4) == 0):
                    print(SofCross.anserrate(ans))"""

                # 学習
                delta_outAf2 = SofCross.back()
                #print(delta_outAf2.shape) -> C * B
                delta_outD = Affine2.SAM_back1(delta_outAf2)
                # print(delta_outS.shape) -> M * B
                delta_outS = dropout.back(delta_outD)
                # print() -> M * B
                delta_outN = relu.back(delta_outS)
                # print(delta_outN.shape) -> M * B
                delta_outP = Bnormal.SAM_back1(delta_outN)
                # print(delta_outAf1.shape) -> M * B
                # print(delta_outP.shape) -> (ch * imgsize) * B
                delta_outC = pooling.back(delta_outP.T)
                _ = Conv.SAM_back1(delta_outC)

                delta_outAf2 = SofCross.back()
                #print(delta_outAf2.shape) -> C * B
                delta_outD = Affine2.SAM_back2(delta_outAf2)
                # print(delta_outS.shape) -> M * B
                delta_outS = dropout.back(delta_outD)
                # print() -> M * B
                delta_outN = relu.back(delta_outS)
                # print(delta_outN.shape) -> M * B
                delta_outP = Bnormal.SAM_back2(delta_outN)
                # print(delta_outAf1.shape) -> M * B
                # print(delta_outP.shape) -> (ch * imgsize) * B
                delta_outC = pooling.back(delta_outP.T)
                _ = Conv.SAM_back2(delta_outC)

            #1epoch終了
            crossE = crossE / epoch
            print(str(i) + "epoch終了", end=":")
            print(crossE)

            correctnum = 0
            #過学習判定
            test = Initialize.Initialize(Xtest, Ytest, self.B, self.C, self.smoothp)
            #テストデータB * 10仕様
            for i in range(Xtest.shape[0] // self.B):
                Batch_img, ans = test.orderselect(i)
                Batch_img = Batch_img.reshape(self.B, 1, imgl, -1)
                # 画像畳み込み
                outC = Conv.prop(Batch_img)
                # プーリング
                outP = pooling.pooling(outC, self.poolw)
                # 形変える
                outP = outP.reshape(outP.shape[0], -1, 1)
                # 正規化
                outN = Bnormal.prop(outP)
                # 中間層
                outS = relu.prop(outN)
                #ドロップアウト
                outD = dropout.test(outS)
                # ~最終層
                outAf2 = Affine2.prop(outD)
                # 最終層
                finout = SofCross.prop(outAf2)   
                correctnum += SofCross.anserrate(ans)
            correctnum = correctnum / 2
            print("contest_rate")
            print(correctnum)
            if(correctnum > precorrectnum): 
                print("更新!!")
                np.savez("./Parameters/CNN-4", Affine2.W, Affine2.b, Bnormal.beta, Bnormal.ganma,
                         Conv.filter_W, Conv.bias)
                np.savez("./Parameters/finout4", finout)
                precorrectnum = correctnum
                if(correctnum > 0.95):
                    print("finish")
                    break

    def test(self):
        # 層に関して
        B = self.B
        M = self.M
        C = self.C
        parameters = np.load("./Parameters/CNN.npz")
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
        Conv = Conv3D.Conv3D(self.ch, self.filw, Xtest.shape[1], B, 1)
        pooling = Pooling.Pooling()
        Affine1 = Affine.Affine(
            self.ch * (imgl // self.poolw) * (imgl // self.poolw), M)
        Bnormal = BatchNormalize.BatchNormalize(M)
        Sigmoid = Activate.Sigmoid()
        Affine2 = Affine.Affine(M, C)
        SofCross = Softmax_cross.Softmax_cross()
        for i in range(epoch):
            Batch_img, ans = Ini.orderselect(i)
            Batch_img = Batch_img.reshape(B, 1, 28, 28)
            # 画像畳み込み
            outC = Conv.test(Batch_img, filter_W, fil_bias)
            # プーリング
            outP = pooling.pooling(outC, self.poolw)
            outP = outP.reshape(outP.shape[0], -1, 1)
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

"""
di = Diversify.Diversify()
X = np.array(mnist.download_and_parse_mnist_file("train-images-idx3-ubyte.gz"))
Xtest = di.expand(X[0].reshape(1, 28, -1))
Imshow.imshow(Xtest)
#"""

print("study -> 0, test -> 1")
a = int(input())
test(a)
#"""