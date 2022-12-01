import numpy as np
import mnist
import matplotlib.pyplot as plt
from pylab import cm
import sys

#変数の用意
#randseed
seed = 10
#中間層の数
M = 2
#最終層の数
C = 10

#シード固定
np.random.seed(seed)

#キーボードの入力受付
print("1~9999")
string_idx = input()
idx = int(string_idx)
#毎回面倒なので一度固定
#idx = 100

#a番目の画像取得
X = mnist.download_and_parse_mnist_file("train-images-idx3-ubyte.gz")
Y = mnist.download_and_parse_mnist_file("train-labels-idx1-ubyte.gz")
before_conv = np.array(X[idx])

#idx番目を表示
#plt.imshow(X[idx], cmap=cm.gray)
#plt.show()

#行列の変形
img_width = before_conv.shape[0]
img_height = before_conv.shape[1]
img_size = before_conv.size
img = before_conv.reshape(img_size, 1)

#ランダムな重みを作成
W1 = np.random.normal(loc = 0, scale = np.sqrt(1/img_size), size=(M, img_size))
b1 = np.random.normal(loc = 0, scale = np.sqrt(1/img_size), size=(M, 1))
W2 = np.random.normal(loc = 0, scale = np.sqrt(1/img_size), size=(C, M))
b2 = np.random.normal(loc = 0, scale = np.sqrt(1/img_size), size=(C, 1))

#中間層への入力
input1 = np.dot(W1, img) + b1
#中間層の出力
output1 = 1/(1 + np.exp(-1 * input1))

#最終層への入力
input2 = np.dot(W2, output1) + b2
#最終層の出力
alpha = input2.max()
sumexp = np.sum(np.exp(input2 - alpha))
output_last = np.exp(input2 - alpha) / sumexp

#尤度最大値を取得
answer = np.argmax(output_last)
print(answer)
