import numpy as np
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from Layer import Affine
from Layer import Activate
from Layer import Softmax_cross
from Layer import Initialize
from Layer import Conv2D
from Layer import Pooling
from Layer import BatchNormalize


def test(Xtest):
    #層に関して
    B = 100
    M = 200
    C = 10
    poolw = 2
    ch = 32
    filw = 5
    parameters = np.load("./Parameters/test.npz")
    W1 = parameters['arr_0']
    W2 = parameters['arr_2']
    b1 = parameters['arr_1']
    b2 = parameters['arr_3']
    normal_beta = parameters['arr_4']
    noraml_ganma = parameters['arr_5']
    filter_W = parameters['arr_6']
    fil_bias = parameters['arr_7']

    Xtest = np.array(Xtest).reshape(-1, 28, 28)
    Ini = Initialize.Initialize(Xtest, None, B, C)
    
    imgl = Xtest.shape[1]

    epoch = Xtest.shape[0] // B
    img_size = Xtest[0].size
    correct_num = 0
    Conv = Conv2D.conv2D(ch, filw, Xtest.shape[1], B)
    pooling = Pooling.Pooling()
    Affine1 = Affine.Affine(ch * (imgl // poolw) * (imgl // poolw), M)
    Bnormal = BatchNormalize.BatchNormalize(M)
    Sigmoid = Activate.Sigmoid()
    Affine2 = Affine.Affine(M, C)
    SofCross = Softmax_cross.Softmax_cross()
    total_expect = np.array([])
    for i in range(epoch):
        Batch_img = Ini.orderselect(i)
        #画像畳み込み
        outC = Conv.test(Batch_img, filter_W, fil_bias)
        #プーリング
        outP = pooling.pooling(outC, poolw)
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

        output_last = np.reshape(finout, (B, C))
        expect = np.argmax(output_last, axis=1)
        total_expect = np.concatenate([total_expect, expect])
    np.savetxt("./contest/predict.txt", total_expect, fmt="%.0f")

list_arr = []

with open("./contest/le4MNIST_X.txt") as f:
        for line in f:
            line = line.rstrip()
            l = line.split()
            arr = list(map(int, l))
            list_arr.append(arr)


test(list_arr)