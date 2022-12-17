import numpy as np
import mnist
import matplotlib.pyplot as plt
from pylab import cm
import sys

#変数の用意
#randseed
seed = 601
#中間層の数
M = 100
#最終層の数
C = 10
#バッチサイズ
B = 100
#シード固定
np.random.seed(seed)

#バッチサイズだけランダムに0~59999を選択
batch_random = np.random.randint(0, 60000, B)


#バッチ画像取得
X = mnist.download_and_parse_mnist_file("train-images-idx3-ubyte.gz")
Y = mnist.download_and_parse_mnist_file("train-labels-idx1-ubyte.gz")
before_conv = np.array(X[batch_random])
#print(before_conv.shape) -> 100 * 28 * 28
#正解を取得
answer = np.array(Y[batch_random])
#正解をone-hot vectorに
onehot = np.zeros((answer.size, 10))
onehot[np.arange(answer.size), answer] = 1


#idx番目を表示
#plt.imshow(X[idx], cmap=cm.gray)
#plt.show()

#行列の変形
img_width = before_conv.shape[1]
img_height = before_conv.shape[2]
img_size = before_conv[1].size
img = before_conv.reshape((B, img_size, 1))

#ランダムな重みを作成
W1 = np.random.normal(loc = 0, scale = np.sqrt(1/img_size), size=(M, img_size))
b1 = np.random.normal(loc = 0, scale = np.sqrt(1/img_size), size=(M, 1))
W2 = np.random.normal(loc = 0, scale = np.sqrt(1/M), size=(C, M))
b2 = np.random.normal(loc = 0, scale = np.sqrt(1/M), size=(C, 1))

#中間層への入力
input1 = np.matmul(W1, img) + b1
#中間層の出力
output1 = 1/(1 + np.exp(-1 * input1))
#print(output1.shape)->100*100*1

#最終層への入力
input2 = np.matmul(W2, output1) + b2
#print(input2.shape) -> 100 * 10 * 1
#最終層の出力
alpha = np.repeat(input2.max(axis = 1), 10, axis= 1).reshape(B, C, 1)
sumexp = np.repeat(np.sum(np.exp(input2 - alpha), axis=1), 10, axis=1).reshape(B, C, 1)
#print(sumexp.shape) -> 100 * 10 * 1
output_last = np.exp(input2 - alpha) / sumexp
output_last = np.reshape(output_last, (100, 10))

#尤度最大値を取得
#expect = np.argmax(output_last, axis=1)
#print(expect.shape) -> 100 * 1

#クロスエントロピー
print((-1/B)* np.sum(onehot * np.log(output_last)))


