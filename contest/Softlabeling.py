import numpy as np
import sys, os
import mnist
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

class solabel():
    def __init__(self) -> None:
        self.trainrate = []
        self.validrate = []

    def re_labeling(self):
        #層に関して
        B = 100
        C = 10
        poolw = 7
        ch = 32
        filw = 5
        phi = 0.5
        #l2
        l2lambda = 0.0005

        #
        M1 = 10
        M2 = 100
        alpha_max = 0.5

        num = 1000
        epoch = 70000 // B

        X_p, Y_p = ConAn.getmnist()
        contestX, contestY = ConAn.get(0, 200)
        test = Initialize.Initialize(
                contestX, contestY, B, C, 0)

        imgl = 28
        M = ch * (imgl // poolw) * (imgl // poolw)

        labeldate = np.copy(X_p)
        abs_ans = np.copy(Y_p)
        abs_label = np.copy(ConAn.onehot(Y_p))
        label = np.copy(ConAn.onehot(Y_p))

        rng = np.random.default_rng()
        ims = Imshow()
        ra = RArg(0.05, 10, np.pi/5, 5/4, 4, np.pi/9, 5)
        Conv = Conv3D.Conv3D(ch, filw, imgl, B,
                             1, l2lambda, opt="MSGD")
        pooling = Pooling.Pooling()
        Bnormal = BatchNormalize.BatchNormalize(M, l2lambda, opt="MSGD")
        relu = Activate.Relu()
        dropout = Dropout.Dropout(phi, B, M)
        Affine2 = Affine.Affine(M, C, l2lambda, "MSGD")
        SofCross = Softmax_cross.Softmax_cross()
        parameters = np.load("./Parameters/contest_fin.npz")
        W2 = parameters['arr_0']
        b2 = parameters['arr_1']
        normal_beta = parameters['arr_2']
        noraml_ganma = parameters['arr_3']
        filter_W = parameters['arr_4']
        fil_bias = parameters['arr_5']
        Affine2.update(W2, b2)
        Bnormal.update(normal_beta, noraml_ganma)
        Conv.update(filter_W, fil_bias)

        bestcorrectrate = 0
        for i in range(num):

            if(i <= M1):
                alpha = 0
            elif(i <= M2):
                alpha = (i - M1) * alpha_max / (M2 - M1)
            else:
                alpha = alpha_max

            terate = 0
            crossE = 0

            for j in range(epoch):
                percent = Decimal(str(j * 100 / epoch)).quantize(Decimal('0'), rounding=ROUND_HALF_UP)
                print('\r' + "with label:" + str(percent) + '%', end="")
                batch_random = rng.choice(labeldate.shape[0], (B,), replace=False) 
                #画像取得       
                img = np.array(labeldate[batch_random])
                #print(before_conv.shape) -> 100 * 28 * 28
                #正解を取得
                ans = abs_ans[batch_random]
                answer = label[batch_random]
                abs_answer = abs_label[batch_random]

                out_Ra = ra.prop2(img, i)
                outC = Conv.prop(out_Ra.reshape(B, 1, imgl, -1))
                # プーリング
                outP = pooling.pooling(outC, poolw)
                # ~中間層
                # 正規化
                outN = Bnormal.prop(outP.reshape(outP.shape[0], -1, 1))
                # 中間層
                outS = relu.prop(outN)
                # DropOut -> 汎化性能が上がる
                outD = dropout.prop(outS)
                # ~最終層
                outAf2 = Affine2.prop(outD)
                # 最終層
                finout = SofCross.prop(outAf2)
                crossE = crossE + SofCross.crossEn(answer)
                terate += SofCross.anserrate(ans)

                #Softlabeling化
                label[batch_random] = alpha * finout + (1 - alpha) * abs_answer

                # 学習
                delta_outAf2 = SofCross.back()
                # print(delta_outAf2.shape) -> C * B
                delta_outD = Affine2.back_l2(delta_outAf2)
                # print(delta_outS.shape) -> M * B
                delta_outS = dropout.back(delta_outD)
                # print() -> M * B
                delta_outN = relu.back(delta_outS)
                # print(delta_outN.shape) -> M * B
                delta_outP = Bnormal.back_l2(delta_outN)
                # print(delta_outAf1.shape) -> M * B
                # print(delta_outP.shape) -> (ch * imgsize) * B
                delta_outC = pooling.back(delta_outP.T)
                _ = Conv.back_l2(delta_outC)
            self.trainrate.append(terate / (j + 1))
            print('\r' + "crossE=" + str(crossE / (j+1)))

            correctnum = 0 
            for k in range(2):
                Batch_img, ans = test.orderselect(k)
                # 画像畳み込み
                outC = Conv.prop(Batch_img.reshape(B, 1, imgl, -1))
                # プーリング
                outP = pooling.pooling(outC, poolw)
                # 形変える
                outP = outP.reshape(outP.shape[0], -1, 1)
                # 正規化
                outN = Bnormal.prop(outP)
                # 中間層
                outS = relu.prop(outN)
                # ドロップアウト
                outD = dropout.test(outS)
                # ~最終層
                outAf2 = Affine2.prop(outD)
                # 最終層
                _ = SofCross.prop(outAf2)
                correctnum += SofCross.anserrate(ans)
            correctnum = correctnum / 2
            print("ansrate=" + str(correctnum))
            if(bestcorrectrate <= correctnum):
                print("更新")
                np.savez("./Parameters/SoftCNN0", Affine2.W, Affine2.b, Bnormal.beta, Bnormal.ganma,
                         Conv.filter_W, Conv.bias)
                bestcorrectrate = correctnum
            self.validrate.append(correctnum)

se = solabel()
try:
    se.re_labeling()
except KeyboardInterrupt:
    l = len(se.teacherrate)
    y = range(l)
    plt.plot(y, se.validrate[0:l], label="contest")
    plt.plot(y, se.trainrate, label="teacher")
    plt.legend()
    plt.show()
    ims = Imshow()