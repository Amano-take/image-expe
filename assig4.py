import numpy as np
import mnist
import matplotlib.pyplot as plt
from pylab import cm
import sys

def sigmoid(x):
    sigmoid_range = 34.53877639410684

    if x <= -sigmoid_range:
        return 1e-15
    if x >= sigmoid_range:
        return 1.0 - 1e-15
    return 1.0 / ( 1.0 + np.exp(-x))

vsigmoid = np.vectorize(sigmoid)

#変数の用意
parameters = np.load("parameter2.npz")
W1 = parameters['arr_0']
W2 = parameters['arr_1']
b1 = parameters['arr_2']
b2 = parameters['arr_3']
#中間層の数
M = 80
#最終層の数
C = 10

X = mnist.download_and_parse_mnist_file("t10k-images-idx3-ubyte.gz")
#Y = mnist.download_and_parse_mnist_file("t10k-labels-idx1-ubyte.gz")
img_width = X.shape[1]
img_height = X.shape[2]
img_size = X[0].size

for i in range(10):
    #キーボードの入力受付
    print("1~9999")
    string_idx = input()
    idx = int(string_idx)

    #a番目の画像取得

    before_conv = np.array(X[idx])

    #idx番目を表示
    plt.imshow(X[idx], cmap=cm.gray)
    plt.show()

    #行列の変形
    img_width = before_conv.shape[0]
    img_height = before_conv.shape[1]
    img_size = before_conv.size
    img = before_conv.reshape(img_size, 1)

    #中間層への入力
    input1 = np.dot(W1, img) + b1
    #中間層の出力
    output1 = vsigmoid(input1)

    #最終層への入力
    input2 = np.dot(W2, output1) + b2
    #最終層の出力
    alpha = input2.max()
    sumexp = np.sum(np.exp(input2 - alpha))
    output_last = np.exp(input2 - alpha) / sumexp

    #尤度最大値を取得
    answer = np.argmax(output_last)
    print("予測:", end = "")
    print(answer) 
    #print("答え:", end = "")
    #print(Y[idx])
