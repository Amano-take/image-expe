import numpy as np
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from Layer import Affine
from Layer import Activate
from Layer import Softmax_cross
from Layer import Initialize
from Layer import Conv3D
from Layer import Pooling
from Layer import BatchNormalize
import mnist

#一番シンプルなもの
class test():

    def __init__(self, a):
        seed = 600
        np.random.seed(seed)
        self.B = 100
        self.M = 90
        self.C = 10
        self.poolw1 = 2
        self.poolw2 = 2
        self.ch1 = 16
        self.ch2 = 4
        self.filw1 = 5
        self.filw2 = 5

        if a == 0:
            self.study()
        else:
            self.test()
        
    def study(self):
        #層に関して
        B = self.B
        M = self.M
        C = self.C
        #画像データ
        X = mnist.download_and_parse_mnist_file("train-images-idx3-ubyte.gz")
        Y = mnist.download_and_parse_mnist_file("train-labels-idx1-ubyte.gz")
        Ini = Initialize.Initialize(X, Y, B, C)
        imgl = X.shape[1]
        #学習に関して
        num = 13
        epoch = X.shape[0] // B
        #crossE初期化
        crossE = 0
        #Layer
        Conv1 = Conv3D.Conv3D(self.ch1, self.filw1, imgl, B, 1)
        Pooling1 = Pooling.Pooling()
        imgl = imgl // self.poolw1
        Conv2 = Conv3D.Conv3D(self.ch2, self.filw2, imgl, B, self.ch1)
        Pooling2 = Pooling.Pooling()
        Affine1 = Affine.Affine(self.ch2 * 7 * 7, M)
        Bnormal = BatchNormalize.BatchNormalize(M)
        Sigmoid = Activate.Sigmoid()
        Affine2 = Affine.Affine(M, C)
        SofCross = Softmax_cross.Softmax_cross()
        for i in range(num):
            precrossE = crossE
            for j in range(epoch):
                Batch_img, onehot = Ini.randomselect()
                #画像畳み込み
                Batch_img = Batch_img.reshape(Batch_img.shape[0], 1, Batch_img.shape[1], -1)
                outC1 = Conv1.prop(Batch_img)
                #プーリング
                outP1 = Pooling1.pooling(outC1, self.poolw1)
                #もう一回
                outC2 = Conv2.prop(outP1)
                outP2 = Pooling2.pooling(outC2, self.poolw2)
                #変換　
                outP2 = outP2.reshape(B, -1, 1)
                #~中間層
                outAf1 = Affine1.prop(outP2)
                #正規化
                outN = Bnormal.prop(outAf1)
                #中間層
                outS = Sigmoid.prop(outN)
                #~最終層
                outAf2 = Affine2.prop(outS)
                #最終層
                finout = SofCross.prop(outAf2)
                crossE = crossE + SofCross.crossEn(onehot)

                #学習
                delta_outAf2 = SofCross.back()
                #print(delta_outAf2.shape) -> C * M
                delta_outS = Affine2.back(delta_outAf2)
                #print(delta_outS.shape) -> M * B
                delta_outN = Sigmoid.back(delta_outS)
                #print(delta_outN.shape) -> M * B
                delta_outAf1 = Bnormal.back(delta_outN)
                #print(delta_outAf1.shape) -> M * B
                delta_outP2 = Affine1.back(delta_outAf1)
                #print(delta_outP.shape) -> (ch * imgsize) * B
                delta_outC2 = Pooling2.back(delta_outP2.T)
                delta_outP1 = Conv2.back(delta_outC2)
                delta_C1 = Pooling1.back(delta_outP1)
                _ = Conv1.back(delta_C1)

            crossE = crossE / epoch
            print(crossE)
            if(crossE < precrossE):
                np.savez("./Parameters/CNN", Affine1.W, Affine1.b, Affine2.W, Affine2.b, Bnormal.beta, Bnormal.ganma,
                                                Conv1.filter_W, Conv1.bias, Conv2.filter_W, Conv2.bias)
            
        print("finish!")

    def test(self):
        #層に関して
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

        Xtest = mnist.download_and_parse_mnist_file("t10k-images-idx3-ubyte.gz")
        Ytest = mnist.download_and_parse_mnist_file("t10k-labels-idx1-ubyte.gz")
        Ini = Initialize.Initialize(Xtest, Ytest, self.B, self.C)
        imgl = Xtest.shape[1]

        epoch = Xtest.shape[0] // self.B
        img_size = Xtest[0].size
        correct_num = 0
        Conv = Conv2D.conv2D(self.ch, self.filw, Xtest.shape[1], B)
        pooling = Pooling.Pooling()
        Affine1 = Affine.Affine(self.ch * (imgl // self.poolw) * (imgl // self.poolw), M)
        Bnormal = BatchNormalize.BatchNormalize(M)
        Sigmoid = Activate.Sigmoid()
        Affine2 = Affine.Affine(M, C)
        SofCross = Softmax_cross.Softmax_cross()
        for i in range(epoch):
            Batch_img, ans = Ini.orderselect(i)
            #画像畳み込み
            outC = Conv.test(Batch_img, filter_W, fil_bias)
            #プーリング
            outP = pooling.pooling(outC, self.poolw)
            #~中間層
            outAf1 = Affine1.test(W1, b1, outP)
            #正規化
            outN = Bnormal.test(outAf1, normal_beta, noraml_ganma)
            #中間層
            outS = Sigmoid.prop(outN)
            #~最終層
            outAf2 = Affine2.test(W2, b2, outS)
            #最終層
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