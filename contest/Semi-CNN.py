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
class semiCNN():
    def __init__(self):
        self.correctrate = []
        self.teacherrate = []

    def study(self):
        #層に関して
        B = 100
        C = 10
        poolw = 7
        ch = 32
        filw = 5
        phi = 0.5
        #l2
        l2lambda = 0.0005
        #semi-
        kpoint = 0.90
        

        num = 1000000

        X_p, Y_p = ConAn.getmnist()
        Xtest = ConAn.gettest()
        contestX, contestY = ConAn.get(0, 200)
        test = Initialize.Initialize(
                contestX, contestY, B, C, 0)

        imgl = 28
        M = ch * (imgl // poolw) * (imgl // poolw)

        labeldate = np.copy(X_p)
        label = np.copy(Y_p)
        nonlabeldate = np.copy(Xtest)

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
        # 途中から学習

        parameters = np.load("./Parameters/semi_CNN0.npz")
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
            
            if(nonlabeldate.shape[0] >= 100):
                nepoch = nonlabeldate.shape[0] // B
                delte_arg = np.empty((0,), dtype=int)
                for k in range(nepoch):
                    percent = Decimal(str(k * 100 / nepoch)).quantize(Decimal('0'), rounding=ROUND_HALF_UP)
                    print('\r' + "non-label:" + str(percent) + '%', end="")
                    batch_random_non = np.arange(k*100, k*100 + 100, dtype=int)
                    img_non = nonlabeldate[batch_random_non]
                    noutC = Conv.prop(img_non.reshape(B, 1, imgl, -1))
                    # プーリング
                    noutP = pooling.pooling(noutC, poolw)
                    # ~中間層
                    # 正規化
                    noutN = Bnormal.prop(noutP.reshape(noutP.shape[0], -1, 1))
                    # 中間層
                    noutS = relu.prop(noutN)
                    # DropOut -> 汎化性能が上がる
                    noutD = dropout.test(noutS)
                    # ~最終層
                    noutAf2 = Affine2.prop(noutD)
                    # 最終層
                    nfinout = SofCross.prop(noutAf2)

                    arg0, arg1, _ = np.where(nfinout > kpoint)
                    labeldate = np.vstack((labeldate, img_non[arg0]))
                    label = np.hstack((label, arg1))
                    #ims.imansshow(nonlabeldate[batch_random_non[arg0]], arg1)
                    delte_arg = np.hstack((delte_arg, batch_random_non[arg0]))
                nonlabeldate = np.delete(nonlabeldate, delte_arg, axis=0)
                    
            else:
                np.savez("./Parameters/semi_CNN1", Affine2.W, Affine2.b, Bnormal.beta, Bnormal.ganma,
                         Conv.filter_W, Conv.bias)
                break
                """nonlabelB = nonlabeldate.shape[0]
                Conv.updateB(nonlabelB)

                noutC = Conv.prop(nonlabeldate.reshape(nonlabelB, 1, imgl, -1))
                # プーリング
                noutP = pooling.pooling(noutC, poolw)
                # ~中間層
                # 正規化
                noutN = Bnormal.prop(noutP.reshape(noutP.shape[0], -1, 1))
                # 中間層
                noutS = relu.prop(noutN)
                # DropOut -> 汎化性能が上がる
                noutD = dropout.prop(noutS)
                # ~最終層
                noutAf2 = Affine2.prop(noutD)
                # 最終層
                nfinout = SofCross.prop(noutAf2)

                arg0, arg1, _ = np.where(nfinout > kpoint)
                labeldate = np.vstack((labeldate, nonlabeldate[arg0]))
                label = np.hstack((label, arg1))
                nonlabeldate = np.delete(nonlabeldate, arg0, axis=0)"""
    
            print('\r' + str(i+1) + "epoch:\033[0K")
            print("nonlabel=" + str(nonlabeldate.shape[0]) + ", withlabel=" + str(labeldate.shape[0]))
            
            correctnum = 0
            for l in range(2):
                Batch_img, ans = test.orderselect(l)
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
                np.savez("./Parameters/semi_CNN1", Affine2.W, Affine2.b, Bnormal.beta, Bnormal.ganma,
                         Conv.filter_W, Conv.bias)
                bestcorrectrate = correctnum
            self.correctrate.append(correctnum)

            crossE = 0
            epoch = labeldate.shape[0] // B
            terate = 0
            for j in range(epoch // 10):
                #ラベル付きで学習
                percent = Decimal(str(j * 1000 / epoch)).quantize(Decimal('0'), rounding=ROUND_HALF_UP)
                print('\r' + "with label:" + str(percent) + '%', end="")
                batch_random = rng.choice(labeldate.shape[0], (B,), replace=False) 
                #画像取得       
                img = np.array(labeldate[batch_random])
                #print(before_conv.shape) -> 100 * 28 * 28
                #正解を取得
                answer = np.array(label[batch_random])
                #正解をone-hot vectorに
                onehot = np.zeros((answer.size, 10))
                onehot[np.arange(answer.size), answer] = 1
                onehot = onehot.reshape(B, C, 1)

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
                finout = finout.reshape(B, -1)
                """
                expected = np.max(finout, axis=1)
                argp = np.where(expected < lpoint)
                nonlabelargp = np.where(batch_random[argp] >= tenum)
                nonlabeldate = np.vstack((nonlabeldate, img[nonlabelargp]))
                labeldate = np.delete(labeldate, batch_random[nonlabelargp], axis=0)
                label = np.delete(label, batch_random[nonlabelargp])"""

                crossE = crossE + SofCross.crossEn(onehot)
                terate += SofCross.anserrate(answer)

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
            self.teacherrate.append(terate / (j + 1))
            print('\r' + "crossE=" + str(crossE / (j+1)))
            
        

            
            
se = semiCNN()
try:
    se.study()
except KeyboardInterrupt:
    l = len(se.teacherrate)
    y = range(l)
    plt.plot(y, se.correctrate[0:l], label="contest")
    plt.plot(y, se.teacherrate, label="teacher")
    plt.legend()
    plt.show()
    ims = Imshow()