import numpy as np
import mnist
import matplotlib.pyplot as plt
from pylab import cm
import sys

class Batch():
    #変数の用意
    #バッチ画像取得
    X = mnist.download_and_parse_mnist_file("train-images-idx3-ubyte.gz")
    Y = mnist.download_and_parse_mnist_file("train-labels-idx1-ubyte.gz")
    
    #randseed
    seed = 601
    #シード固定
    np.random.seed(seed)

    #中間層の数
    M = 80
    #最終層の数
    C = 10
    #バッチサイズ
    B = 100
    #学習率
    my = 0.01
    #エポック
    epoch = X.shape[0] // B
    #エポック繰り返し
    num = 100
    
    def __init__(self, i):
        self.ganma = 1
        self.beta = 0
        
        if(i == 0):
            self.study()
        else:
            self.test()
        

    #活性化関数
    @staticmethod
    @np.vectorize
    def vsigmoid(x):
        sigmoid_range = 34.53877639410684

        if x <= -sigmoid_range:
            return 1e-15
        if x >= sigmoid_range:
            return 1.0 - 1e-15
        return 1.0 / ( 1.0 + np.exp(-x))
    
    def normalize(self, x):
        #0次元方向をバッチと想定
        # x = B * ...
        microB = np.sum(x, axis = 0) / x.shape[0]
        sigmaB = np.sum((x - microB) ** 2, axis = 0) / x.shape[0]
        normalize_x = (x - microB) / np.sqrt(sigmaB + 1e-12)
        y = self.ganma * normalize_x + self.beta
        #yをreturn
        return y

    def study(self):
        B = Batch.B
        M = Batch.M
        C = Batch.C
        img_size = Batch.img_size
        num = Batch.num
        msk_num = Batch.msk_num
        my = Batch.my

        #クロスエントロピー
        crossE = 0
        #ランダムな重みを作成
        W1 = np.random.normal(loc = 0, scale = np.sqrt(1/img_size), size=(M, img_size))
        b1 = np.random.normal(loc = 0, scale = np.sqrt(1/img_size), size=(M, 1))
        W2 = np.random.normal(loc = 0, scale = np.sqrt(1/img_size), size=(C, M))
        b2 = np.random.normal(loc = 0, scale = np.sqrt(1/img_size), size=(C, 1))
        for i in range(num):
            #1エポック
            previouscrossE = crossE
            for j in range(Batch.epoch):
                #バッチサイズだけランダムに0~59999を選択
                batch_random = np.random.randint(0, 60000, B) 
                #画像取得       
                before_conv = np.array(Batch.X[batch_random])
                #print(before_conv.shape) -> 100 * 28 * 28
                #正解を取得
                answer = np.array(Batch.Y[batch_random])
                #正解をone-hot vectorに
                onehot = np.zeros((answer.size, 10))
                onehot[np.arange(answer.size), answer] = 1
                onehot = onehot.reshape(B, C, 1)

                #行列の変形
                img = before_conv.reshape((B, img_size, 1))

                #(中間層）活性化層に入る前に正規化 ... 雑に調べた結果活性化層の前に正規化層を入れるっぽい...?
                img = self.normalize(img)
                
                #中間層への入力
                input1 = np.matmul(W1, img) + b1
                #中間層の出力 (mask)
                #オーバーフロー対策済み
                output1 = Batch.vsigmoid(input1)

                #最終層への入力
                input2 = np.matmul(W2, output1) + b2
                #print(input2.shape) -> 100 * 10 * 1
                #最終層の出力
                alpha = np.repeat(input2.max(axis = 1), C, axis= 1).reshape(B, C, 1)
                sumexp = np.repeat(np.sum(np.exp(input2 - alpha), axis=1), 10, axis=1).reshape(B, C, 1)
                #print(sumexp.shape) -> 100 * 10 * 1
                output_last = np.exp(input2 - alpha) / sumexp

                #クロスエントロピー
                crossE += (-1/B)* np.sum(onehot * np.log(output_last))

                #微分
                #ソフト+クロスエントロピー
                delta_b = ((output_last - onehot)/B).reshape(B, C).T
                    #print(delta_b.shape) -> 10, 100
                #中間層~最終層
                delta_y1 = np.dot(W2.T, delta_b)
                    #print(delta_y1.shape) ->  * 100
                delta_W2 = np.dot(delta_b, output1.reshape(B, M))
                delta_b2 = np.sum(delta_b, axis = 1).reshape(C, 1)
                #シグモイド関数
                delta_a_prime = delta_y1 * (1 - output1.reshape(B, M).T) * output1.reshape(B, M).T
                    #print(delta_a.shape) -> C * B
                #正規化部分
                delta_xi_head = delta_a_prime * self.ganma
                delta_sigmaB2 = 
                #入力層~中間層
                delta_x = np.matmul(W1.T, delta_a)
                delta_W1 = np.matmul(delta_a, img.reshape(B, img_size))
                delta_b1 = np.sum(delta_a, axis=1).reshape(M, 1)

                #パラメータ更新
                W1 = W1 - my * delta_W1
                b1 = b1 - my * delta_b1
                W2 = W2 - my * delta_W2
                b2 = b2 - my * delta_b2
            crossE = crossE / Batch.epoch
            print(crossE)

            if(crossE < previouscrossE):
                np.savez("parameter_Batch", W1, W2, b1, b2)
    
    def test(self):
        parameters = np.load("parameter_Batch.npz")
        W1 = parameters['arr_0']
        W2 = parameters['arr_1']
        b1 = parameters['arr_2']
        b2 = parameters['arr_3']

        Xtest = mnist.download_and_parse_mnist_file("t10k-images-idx3-ubyte.gz")
        Ytest = mnist.download_and_parse_mnist_file("t10k-labels-idx1-ubyte.gz")
        #Xtest = mnist.download_and_parse_mnist_file("train-images-idx3-ubyte.gz")
        #Ytest = mnist.download_and_parse_mnist_file("train-labels-idx1-ubyte.gz")
        M = Batch.M
        C = Batch.C
        before_conv = np.array(Xtest)
        B = before_conv.shape[0]
        answer = np.array(Ytest)
        img_size = before_conv[1].size
        img = before_conv.reshape((B, img_size, 1))

        #中間層への入力
        input1 = np.matmul(W1, img) + b1
        #中間層の出力
        output1 = Batch.vsigmoid(input1) * (1 - Batch.msk_num/M)
        #print(output1.shape)->100*100*1

        #最終層への入力
        input2 = np.matmul(W2, output1) + b2
        #print(input2.shape) -> 100 * 10 * 1
        #最終層の出力
        alpha = np.repeat(input2.max(axis = 1), 10, axis= 1).reshape(B, C, 1)
        sumexp = np.repeat(np.sum(np.exp(input2 - alpha), axis=1), 10, axis=1).reshape(B, C, 1)
        #print(sumexp.shape) -> 100 * 10 * 1
        output_last = np.exp(input2 - alpha) / sumexp
        output_last = np.reshape(output_last, (B, C))

        #尤度最大値を取得
        mis_list = []
        expect = np.argmax(output_last, axis=1)
        num_correct = 0
        for i, ans in enumerate(answer):
            if ans == expect[i]:
                num_correct = num_correct + 1
            else:
                mis_list.append(i)
        print(num_correct* 100 / B)

print("study -> 0, test -> 1")
a = int(input())
Batch(a)




