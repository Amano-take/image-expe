import numpy as np
import mnist
import matplotlib.pyplot as plt
from pylab import cm
import sys

#変数の用意
#randseed
seed = 209
#中間層の数
M = 80
#最終層の数
C = 10
#バッチサイズ
B = 100
#学習率
my = 0.01
#シード固定
np.random.seed(seed)
#バッチ画像取得
X = mnist.download_and_parse_mnist_file("train-images-idx3-ubyte.gz")
Y = mnist.download_and_parse_mnist_file("train-labels-idx1-ubyte.gz")
X_test = np.array(mnist.download_and_parse_mnist_file("t10k-images-idx3-ubyte.gz"))
Y_test = np.array(mnist.download_and_parse_mnist_file("t10k-labels-idx1-ubyte.gz"))
#画像の形を取得
img_width = X.shape[1]
img_height = X.shape[2]
img_size = X[0].size
#エポック
epoch = X.shape[0] // B
#エポック繰り返し
num = 30
val_rate = []
train_rate = []
val_loss = []
train_loss = []

#ランダムな重みを作成
W1 = np.random.normal(loc = 0, scale = np.sqrt(1/img_size), size=(M, img_size))
b1 = np.random.normal(loc = 0, scale = np.sqrt(1/img_size), size=(M, 1))
W2 = np.random.normal(loc = 0, scale = np.sqrt(1/M), size=(C, M))
b2 = np.random.normal(loc = 0, scale = np.sqrt(1/M), size=(C, 1))

#ロードする場合
"""parameters = np.load("./Parameters/parameter.npz")
W1 = parameters['arr_0']
W2 = parameters['arr_1']
b1 = parameters['arr_2']
b2 = parameters['arr_3']"""


for i in range(10000):
    #1エポック
    crossE = 0
    ansnum = 0 
    crossE_val = 0
    ansnum_val = 0

    for j in range(epoch):
        #バッチサイズだけランダムに0~59999を選択
        batch_random = np.random.randint(0, 60000, B)
        #画像取得       
        before_conv = np.array(X[batch_random]) / 255
        #print(before_conv.shape) -> 100 * 28 * 28
        #正解を取得
        answer = np.array(Y[batch_random])
        #正解をone-hot vectorに
        onehot = np.zeros((answer.size, 10))
        onehot[np.arange(answer.size), answer] = 1
        onehot = onehot.reshape(B, C, 1)

        #行列の変形
        img = before_conv.reshape((B, img_size, 1))
        
        #中間層への入力
        input1 = np.matmul(W1, img) + b1
        #中間層の出力
        
        output1 = 1/(1 + np.exp(-1 * input1))

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
        expect = np.argmax(output_last.reshape(B, -1), axis= 1)
        ansnum += np.count_nonzero(np.equal(expect.flatten(), answer.flatten()))

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
        delta_a = delta_y1 * (1 - output1.reshape(B, M).T) * output1.reshape(B, M).T
            #print(delta_a.shape) -> C * B
        #入力層~中間層
        delta_x = np.matmul(W1.T, delta_a)
        delta_W1 = np.matmul(delta_a, img.reshape(B, img_size))
        delta_b1 = np.sum(delta_a, axis=1).reshape(M, 1)

        #パラメータ更新
        W1 = W1 - my * delta_W1
        b1 = b1 - my * delta_b1
        W2 = W2 - my * delta_W2
        b2 = b2 - my * delta_b2
    print(crossE / epoch)
    train_loss.append(crossE / epoch)
    train_rate.append(ansnum / (epoch * B))

    for k in range(X_test.shape[0] // B):
        before_conv = X_test[k*100:k*100 + 100] / 255
        answer = Y_test[k*100:k*100 + 100]
        onehot = np.zeros((answer.size, 10))
        onehot[np.arange(answer.size), answer] = 1
        onehot = onehot.reshape(B, C, 1)

        #行列の変形
        img = before_conv.reshape((B, img_size, 1))
        
        #中間層への入力
        input1 = np.matmul(W1, img) + b1
        #中間層の出力
        
        output1 = 1/(1 + np.exp(-1 * input1))

        #最終層への入力
        input2 = np.matmul(W2, output1) + b2
        #print(input2.shape) -> 100 * 10 * 1
        #最終層の出力
        alpha = np.repeat(input2.max(axis = 1), C, axis= 1).reshape(B, C, 1)
        sumexp = np.repeat(np.sum(np.exp(input2 - alpha), axis=1), 10, axis=1).reshape(B, C, 1)
        #print(sumexp.shape) -> 100 * 10 * 1
        output_last = np.exp(input2 - alpha) / sumexp

        #クロスエントロピー
        crossE_val += (-1/B)* np.sum(onehot * np.log(output_last))
        expect = np.argmax(output_last.reshape(B, -1), axis= 1)
        ansnum_val += np.count_nonzero(np.equal(expect.flatten(), answer.flatten()))

    val_loss.append(crossE_val / (k + 1))
    val_rate.append(ansnum_val / ((k + 1) * B))

y = len(val_loss)
plt.plot(range(y), train_loss[0:y], label="train_loss")
plt.plot(range(y), val_loss, label="validation_loss")
plt.legend()
plt.show()

plt.plot(range(y), train_rate[0:y], label="train_rate")
plt.plot(range(y), val_rate, label="validation_rate")
plt.legend()
plt.show()
np.savez("./Parameters/parameter_assig3", W1, W2, b1, b2)



