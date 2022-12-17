import numpy as np
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
#from Advanced import CNN
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


def test():
    #層に関して
    B = 100
    C = 10
    poolw = 7
    ch = 32
    filw = 5
    phi = 0.5

    parameters = np.load("./Parameters/CNN-2.npz")
    W2 = parameters['arr_0']
    b2 = parameters['arr_1']
    normal_beta = parameters['arr_2']
    noraml_ganma = parameters['arr_3']
    filter_W = parameters['arr_4']
    fil_bias = parameters['arr_5']

    Xtest = ConAn.gettest()
    Ini = Initialize.Initialize(Xtest, None, B, C, 0)
    
    imgl = Xtest.shape[1]
    M = ch * (imgl // poolw) * (imgl // poolw)

    epoch = Xtest.shape[0] // B
    img_size = Xtest[0].size

    Conv = Conv3D.Conv3D(ch, filw, imgl, B, 1)
    pooling = Pooling.Pooling()
    Bnormal = BatchNormalize.BatchNormalize(M)
    relu = Activate.Relu()
    dropout = Dropout.Dropout(phi, B, M)
    Affine2 = Affine.Affine(M, C)
    SofCross = Softmax_cross.Softmax_cross()
    total_expect = np.array([])
    for i in range(epoch):
        Batch_img = Ini.orderselect_test(i)
        Batch_img = Batch_img.reshape(B, 1, 28, 28)
        #画像畳み込み
        outC = Conv.test(Batch_img, filter_W, fil_bias)
        #プーリング
        outP = pooling.pooling(outC, poolw)
        outP = outP.reshape(outP.shape[0], -1, 1)
        #正規化
        outN = Bnormal.test(outP, normal_beta, noraml_ganma)
        #中間層
        outS = relu.prop(outN)
        #ドロップアウトテスト
        outD = dropout.test(outS)
        #~最終層
        outAf2 = Affine2.test(W2, b2, outD)
        #最終層
        finout = SofCross.prop(outAf2)

        output_last = np.reshape(finout, (B, C))
        expect = np.argmax(output_last, axis=1)
        total_expect = np.concatenate([total_expect, expect])
    np.savetxt("./contest/predict.txt", total_expect, fmt="%.0f")
test()