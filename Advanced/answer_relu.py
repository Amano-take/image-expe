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

parameters = np.load("./Advanced/parameter_relu.npz")
W1 = parameters['arr_0']
W2 = parameters['arr_1']
b1 = parameters['arr_2']
b2 = parameters['arr_3']

#中間層の数
M = 80
#最終層の数
C = 10


#バッチ画像取得
X = mnist.download_and_parse_mnist_file("t10k-images-idx3-ubyte.gz")
Y = mnist.download_and_parse_mnist_file("t10k-labels-idx1-ubyte.gz")
before_conv = np.array(X)
#print(before_conv.shape) -> 100 * 28 * 28
#正解を取得
answer = np.array(Y)
B = before_conv.shape[0]


#行列の変形
img_width = before_conv.shape[1]
img_height = before_conv.shape[2]
img_size = before_conv[1].size
img = before_conv.reshape((B, img_size, 1))

#中間層への入力
input1 = np.matmul(W1, img) + b1
#中間層の出力
output1 = np.where(input1 <= 0, 0, input1)
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
#print(expect.shape) -> 100 * 1









