import numpy as np
import mnist
import matplotlib.pyplot as plt
from pylab import cm
import sys
#import Advanced.batch_nomal 

eps = 1e-12

def conv2D_prime(x, W, R, b):
    #filterの形で持たずに常にWで持ってもよい
    r = R // 2
    x_prime = np.pad(x, [(0,), (r,), (r,)], "constant")
    X = x2X(x_prime, R)
    Y = np.dot(W, X)  + b
    return X, Y

def x2X(x, R):
    B, x_length, x_width = x.shape
    dx = x_length - R + 1
    dy = x_width - R + 1
    altx = np.zeros((B, R, R, dx, dy))
    for i in range(R):
        for j in range(R):
            altx[:, i, j, :, :] = x[:, i:i+dx, j:j+dy]
    return altx.transpose(1, 2, 0, 3, 4).reshape(R*R, dx*dy*B)

@np.vectorize
def vsigmoid(x):
    sigmoid_range = 34.53877639410684

    if x <= -sigmoid_range:
        return 1e-15
    if x >= sigmoid_range:
        return 1.0 - 1e-15
    return 1.0 / ( 1.0 + np.exp(-x))

def normalize_test(x, ganma, beta):
    microB = np.sum(x, axis = 0) / x.shape[0]
    sigmaB = np.sum((x - microB) ** 2, axis = 0) / x.shape[0]
    y = ganma * np.power(sigmaB + eps, -1/2) * x + ( beta - ganma * microB * np.power(sigmaB + eps, -1/2))

    #yをreturn
    return y

def normalize_one(x):
        #x = B * 784 * 1
        microx = (np.sum(x, axis = 1) / x.shape[1]).reshape(x.shape[0], 1, 1)
        sigmax = (np.sum((x - microx) ** 2, axis = 1) / x.shape[0]).reshape(x.shape[0], 1, 1)
        return (x - microx) / np.sqrt(sigmax)


parameters = np.load("./Parameters/conv_Adam.npz")
W1 = parameters['arr_0']
W2 = parameters['arr_1']
b1 = parameters['arr_2']
b2 = parameters['arr_3']
beta = parameters['arr_4']
ganma = parameters['arr_5']
W_filter = parameters['arr_6']
b_filter = parameters['arr_7']

list_arr = []

with open("./contest/le4MNIST_X.txt") as f:
        for line in f:
            line = line.rstrip()
            l = line.split()
            arr = list(map(int, l))
            list_arr.append(arr)

list_ans = []

with open("./contest/answer.txt") as f:
        for line in f:
            line = line.rstrip()
            arr = list(map(int, line))
            list_ans.append(arr)


Xtest = np.array(list_arr[0:100])
Ytest = np.array(list_ans)
#Xtest = mnist.download_and_parse_mnist_file("t10k-images-idx3-ubyte.gz")
#Ytest = mnist.download_and_parse_mnist_file("t10k-labels-idx1-ubyte.gz")
#Xtest = mnist.download_and_parse_mnist_file("train-images-idx3-ubyte.gz")
#Ytest = mnist.download_and_parse_mnist_file("train-labels-idx1-ubyte.gz")
M = 80
C = 10
before_conv = np.array(Xtest)
B = before_conv.shape[0]
img_size = before_conv.shape[1]
img = before_conv.reshape((B, img_size, 1))
img = normalize_one(img)
_, img = conv2D_prime(img, W_filter, 3, b_filter)
img = img.reshape(B, img_size, 1)

#中間層への入力
input1_prime = np.matmul(W1, img) + b1
input1 = normalize_test(input1_prime, ganma, beta)
#中間層の出力
output1 = vsigmoid(input1)

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
expect = np.argmax(output_last, axis=1)
#np.savetxt("./contest/predict_normalize.txt", expect, fmt="%.0f")


num_correct = 0
for i, an in enumerate(Ytest):
    if an == expect[i]:
        num_correct += 1
print(num_correct  * 100 / B)


