import numpy as np
import mnist
import matplotlib.pyplot as plt
from pylab import cm
import sys


class Batch():
    # 変数の用意
    # バッチ画像取得
    X = np.array(mnist.download_and_parse_mnist_file("train-images-idx3-ubyte.gz"))
    Y = np.array(mnist.download_and_parse_mnist_file("train-labels-idx1-ubyte.gz"))
    img_size = X[0].size
    imgr = X[0].shape[0]

    # randseed
    seed = 601
    # シード固定
    np.random.seed(seed)

    # 中間層の数
    M = 80
    # 最終層の数
    C = 10
    # バッチサイズ
    B = 100
    # 学習率
    my = 0.001
    # エポック
    epoch = X.shape[0] // B
    # エポック繰り返し
    num = 13
    #Adam
    alpha_para = 0.001
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    #filterR
    filterR = 3
    filterK = 10

    def __init__(self, i):
        img_size = Batch.img_size
        imgr = Batch.imgr
        M = Batch.M
        C = Batch.C

        self.ganma = np.ones(Batch.M).reshape(Batch.M, 1)
        self.beta = np.zeros(Batch.M).reshape(Batch.M, 1)
        self.W1 = np.random.normal(loc=0, scale=np.sqrt(
            1/img_size), size=(M, img_size))
        self.b1 = np.random.normal(loc=0, scale=np.sqrt(1/img_size), size=(M, 1))
        self.W2 = np.random.normal(loc=0, scale=np.sqrt(1/img_size), size=(C, M))
        self.b2 = np.random.normal(loc=0, scale=np.sqrt(1/img_size), size=(C, 1))
        self.W_filter = np.random.normal(loc=0, scale=0.01, size=(Batch.filterK, Batch.filterR * Batch.filterR))
        self.b_filter = np.repeat(np.random.normal(loc=0, scale=0.01, size=(Batch.filterK, 1)), imgr*imgr*Batch.B, axis=1)

        if (i == 0):
            self.study()
        else:
            self.test()

    # 活性化関数

    @staticmethod
    @np.vectorize
    def vsigmoid(x):
        sigmoid_range = 34.53877639410684

        if x <= -sigmoid_range:
            return 1e-15
        if x >= sigmoid_range:
            return 1.0 - 1e-15
        return 1.0 / (1.0 + np.exp(-x))

    def normalize(self, x):
        # 0次元方向をバッチと想定
        # x = B * ...
        self.microB = np.sum(x, axis=0) / x.shape[0]
        self.sigmaB = np.sum((x - self.microB) ** 2, axis=0) / x.shape[0]
        self.normalize_x = (x - self.microB) / np.sqrt(self.sigmaB + Batch.eps)
        self.y = self.ganma * self.normalize_x + self.beta

        # yをreturn
        return self.y

    def normalize_one(self, x):
        #x = B * 784 * 1
        microx = (np.sum(x, axis=1) / x.shape[1]).reshape(x.shape[0], 1, 1)
        sigmax = (np.sum((x - microx) ** 2, axis=1) /
                  x.shape[0]).reshape(x.shape[0], 1, 1)
        return (x - microx) / np.sqrt(sigmax)

    def conv2D(self, x, filter, b):
        #A5のch=1の時に限定（ch=3は./3Dにあります）
        #filter = K * R * Rを想定, Rは奇数を想定
        #教科書通りに定義
        K = filter.shape[0]
        R = filter.shape[1]
        r = R // 2
        W = self.fil2W(filter)
        x_prime = np.pad(x, [(0,), (r,), (r,)], "constant")
        X = self.x2X(x_prime, R)
        Y = np.dot(W, X)  + b
        return X, Y
    
    def conv2D_prime(self, x, W, R, b):
        #filterの形で持たずに常にWで持ってもよい
        r = R // 2
        x_prime = np.pad(x, [(0,), (r,), (r,)], "constant")
        X = self.x2X(x_prime, R)
        Y = np.dot(W, X)  + b
        return X, Y
    
    def x2X(self, x, R):
        B, x_length, x_width = x.shape
        dx = x_length - R + 1
        dy = x_width - R + 1
        altx = np.zeros((B, R, R, dx, dy))
        for i in range(R):
            for j in range(R):
                altx[:, i, j, :, :] = x[:, i:i+dx, j:j+dy]
        return altx.transpose(1, 2, 0, 3, 4).reshape(R*R, dx*dy*B)


    def fil2W(self, filter):
        K, R, _ = filter.shape
        return filter.reshape(K, R*R)
        
    # パラメータ更新用関数軍
    def Adam(self, w_b, delta, t, m, v):
        t = t + 1
        m = Batch.beta1 * m + (1 - Batch.beta1) * delta
        v = Batch.beta2 * v + (1 - Batch.beta2) * delta * delta
        m_head = m / (1 - np.power(Batch.beta1, t))
        v_head = v / (1 - np.power(Batch.beta2, t))
        W = w_b - Batch.alpha_para * m_head / \
            (np.power(v_head, 1/2) + Batch.eps)
        return W, m, v, t

    def Adam_

    def study(self):
        B = Batch.B
        M = Batch.M
        C = Batch.C
        img_size = Batch.img_size
        _, imglength, imgwidth = Batch.X.shape
        num = Batch.num
        epoch = Batch.epoch
        my = Batch.my
        dx = imglength - Batch.filterR + 1
        dy = imglength - Batch.filterR + 1

        # クロスエントロピー
        crossE = 0
        # ランダムな重みを作成
        

        # 学習用の初期値
        t = 0
        mw1 = 0
        vw1 = 0
        mw2 = 0
        vw2 = 0
        mb1 = 0
        vb1 = 0
        mb2 = 0
        vb2 = 0
        mbeta = 0
        vbeta = 0
        mganma = 0
        vganma = 0
        mfilW = 0
        vfilW = 0
        mfilb = 0
        vfilb = 0

        for i in range(num):
            # 1エポック
            previouscrossE = crossE
            for j in range(epoch):
                # バッチサイズだけランダムに0~59999を選択
                batch_random = np.random.randint(0, 60000, B)
                # 画像取得
                before_conv = np.array(Batch.X[batch_random])
                # print(before_conv.shape) -> 100 * 28 * 28
                # 正解を取得
                answer = np.array(Batch.Y[batch_random])
                # 正解をone-hot vectorに
                onehot = np.zeros((answer.size, 10))
                onehot[np.arange(answer.size), answer] = 1
                onehot = onehot.reshape(B, C, 1)

                # 行列の変形
                img = before_conv.reshape((B, img_size, 1))
                # 正規化
                img = self.normalize_one(img)
                #畳み込み
                arX, img = self.conv2D_prime(img, W_filter, Batch.filterR, b_filter)
                img = img.reshape(B, img_size, 1)

                # 中間層への入力
                input1_prime = np.matmul(W1, img) + b1
                input1 = self.normalize(input1_prime)
                # print(input1.shape) -> B * M * 1

                # オーバーフロー対策済み
                output1 = Batch.vsigmoid(input1)

                # 最終層への入力
                input2 = np.matmul(W2, output1) + b2
                # print(input2.shape) -> B * C * 1
                # 最終層の出力
                alpha = np.repeat(input2.max(axis=1), C,
                                  axis=1).reshape(B, C, 1)
                sumexp = np.repeat(
                    np.sum(np.exp(input2 - alpha), axis=1), 10, axis=1).reshape(B, C, 1)
                # print(sumexp.shape) -> 100 * 10 * 1
                output_last = np.exp(input2 - alpha) / sumexp

                # クロスエントロピー
                crossE += (-1/B) * np.sum(onehot * np.log(output_last))

                # 微分
                # ソフト+クロスエントロピー
                delta_b = ((output_last - onehot)/B).reshape(B, C).T
                # print(delta_b.shape) -> C * B
                # 中間層~最終層
                delta_y1 = np.dot(W2.T, delta_b)
                # print(delta_y1.shape) -> M * B
                delta_W2 = np.dot(delta_b, output1.reshape(B, M))
                delta_b2 = np.sum(delta_b, axis=1).reshape(C, 1)
                # シグモイド関数
                delta_a_prime = delta_y1 * \
                    (1 - output1.reshape(B, M).T) * output1.reshape(B, M).T
                # normalize
                delta_xi_head = delta_a_prime * self.ganma
                delta_sigmaB2 = np.sum(delta_xi_head * (input1_prime.reshape(B, M).T - self.microB)
                                       * (-1/2) * np.power(self.sigmaB + Batch.eps, -3/2), axis=1).reshape(M, 1)
                delta_microB = np.sum(delta_xi_head * (-1 / np.power(self.sigmaB + Batch.eps, 1/2)), axis=1).reshape(
                    M, 1) + (-2) * delta_sigmaB2 * np.sum(input1_prime.reshape(B, M).T - self.microB, axis=1).reshape(M, 1)
                delta_a = delta_xi_head * np.power(self.sigmaB + Batch.eps, -1/2) + delta_sigmaB2 * 2 * (
                    input1_prime.reshape(B, M).T - self.microB) / B + delta_microB / B
                delta_ganma = np.sum(
                    delta_a_prime * self.normalize_x.reshape(B, M).T, axis=1).reshape(M, 1)
                delta_beta = np.sum(delta_a_prime, axis=1).reshape(M, 1)
                # 入力層~中間層
                delta_x = np.matmul(W1.T, delta_a)
                delta_W1 = np.matmul(delta_a, img.reshape(B, img_size))
                delta_b1 = np.sum(delta_a, axis=1).reshape(M, 1)
                #畳み込み層
                delta_x = delta_x.reshape(1, -1)
                delta_filter_x = np.matmul(W_filter.T, delta_x)
                delta_filter_W = np.matmul(delta_x, arX.T)
                delta_filter_b = np.sum(delta_x, axis=0)
                # パラメータ更新
                W1_para = self.Adam(W1, delta_W1, t, mw1, vw1)
                W1 = W1_para[0]
                mw1 = W1_para[1]
                vw1 = W1_para[2]
                W2_para = self.Adam(W2, delta_W2, t, mw2, vw2)
                W2 = W2_para[0]
                mw2 = W2_para[1]
                vw2 = W2_para[2]
                b1_para = self.Adam(b1, delta_b1, t, mb1, vb1)
                b1 = b1_para[0]
                mb1 = b1_para[1]
                vb1 = b1_para[2]
                b2_para = self.Adam(b2, delta_b2, t, mb2, vb2)
                b2 = b2_para[0]
                mb2 = b2_para[1]
                vb2 = b2_para[2]
                beta_para = self.Adam(self.beta, delta_beta, t, mbeta, vbeta)
                self.beta = beta_para[0]
                mbeta = beta_para[1]
                vbeta = beta_para[2]
                ganma_para = self.Adam(
                    self.ganma, delta_ganma, t, mganma, vganma)
                self.ganma = ganma_para[0]
                mganma = ganma_para[1]
                vganma = ganma_para[2]
                W_filter_para = self.Adam(W_filter, delta_filter_W, t, mfilW, vfilW)
                W_filter = W_filter_para[0]
                mfilW = W_filter_para[1]
                vfilW = W_filter_para[2]
                b_filter_para = self.Adam(b_filter, delta_filter_b, t, mfilb, vfilb)
                b_filter = b_filter_para[0]
                mfilb = b_filter_para[1]
                vfilb = b_filter_para[2]
                t = b_filter_para[3]

            crossE = crossE / Batch.epoch
            print(crossE)

            if (crossE < previouscrossE):
                np.savez("./Parameters/conv_Adam", W1,
                         W2, b1, b2, self.beta, self.ganma)

    def normalize_test(self, x, ganma, beta):
        microB = np.sum(x, axis=0) / x.shape[0]
        sigmaB = np.sum((x - microB) ** 2, axis=0) / x.shape[0]
        y = ganma * np.power(sigmaB + Batch.eps, -1/2) * x + \
            (beta - ganma * microB * np.power(sigmaB + Batch.eps, -1/2))

        # yをreturn
        return y

    def test(self):
        parameters = np.load("./Parameters/conv_Adam.npz")
        W1 = parameters['arr_0']
        W2 = parameters['arr_1']
        b1 = parameters['arr_2']
        b2 = parameters['arr_3']
        beta = parameters['arr_4']
        ganma = parameters['arr_5']

        Xtest = mnist.download_and_parse_mnist_file(
            "t10k-images-idx3-ubyte.gz")
        Ytest = mnist.download_and_parse_mnist_file(
            "t10k-labels-idx1-ubyte.gz")
        #Xtest = mnist.download_and_parse_mnist_file("train-images-idx3-ubyte.gz")
        #Ytest = mnist.download_and_parse_mnist_file("train-labels-idx1-ubyte.gz")
        answer = np.array(Ytest)
        Xtest = np.array(Xtest)
        M = Batch.M
        C = Batch.C
        before_conv = np.array(Xtest)
        B = before_conv.shape[0]
        #answer = np.array(Ytest)
        img_size = before_conv[1].size
        img = before_conv.reshape((B, img_size, 1))
        img = self.normalize_one(img)

        # 中間層への入力
        input1_prime = np.matmul(W1, img) + b1
        input1 = self.normalize_test(input1_prime, ganma, beta)
        # 中間層の出力
        output1 = Batch.vsigmoid(input1)

        # 最終層への入力
        input2 = np.matmul(W2, output1) + b2
        # print(input2.shape) -> 100 * 10 * 1
        # 最終層の出力
        alpha = np.repeat(input2.max(axis=1), 10, axis=1).reshape(B, C, 1)
        sumexp = np.repeat(np.sum(np.exp(input2 - alpha),
                           axis=1), 10, axis=1).reshape(B, C, 1)
        # print(sumexp.shape) -> 100 * 10 * 1
        output_last = np.exp(input2 - alpha) / sumexp
        output_last = np.reshape(output_last, (B, C))

        # 尤度最大値を取得
        mis_list = []
        expect = np.argmax(output_last, axis=1)
        #np.savetxt("./contest/answer.txt", expect, fmt="%.0f")

        num_correct = 0
        for i, ans in enumerate(Ytest):
            if ans == expect[i]:
                num_correct = num_correct + 1
            else:
                mis_list.append(i)
        print(num_correct * 100 / B)


print("study -> 0, test -> 1")
a = int(input())
Batch(a)
