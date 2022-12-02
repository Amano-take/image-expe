import numpy as np
import mnist
import matplotlib.pyplot as plt
from pylab import cm
import pickle

def unpickle(file):
    with open(file, "rb") as fo:
        dict = pickle.load(fo,encoding='bytes')
    key_list = list(dict.keys())
    X = np.array(dict[key_list[2]])
    X = X.reshape((X.shape[0],3,32,32))
    Y = np.array(dict[key_list[1]])
    return X,Y

def convolution_2d(x, filter):
    #xは二次元配列, filter.shape -> r * r
    R = filter.shape[0]
    r = R // 2
    #パディング
    x_prime = np.pad(x, r, "constant")
    return np.dot(filter.reshape(-1), im2col_2d(x_prime, R)).reshape(x.shape)
    
def im2col_2d(x, fsize):
    #fsizeは奇数を期待しています。
    x_size = x.shape[0]
    #高速化アルゴリズムがどうしても思い浮かばなかったので、以下サイト参照しました
    #https://qiita.com/kuroitu/items/35d7b5a4bde470f69570
    o = x_size - fsize + 1
    col = np.empty((fsize, fsize, o, o))
    for h in range(fsize):
        for w in range(fsize):
            col[h, w, :, :] = x[h : h + o, w : w + o]
    return col.reshape(fsize * fsize, o * o)

def convolution(x, filter, B, C):
    #x = B * ch * w * w, filter = filter_num * ch * r * r
    R = filter.shape[0]
    r = R // 2
    x_prime = np.pad(x_prime, [0, 0, r, r], "constant")
    #im2col
    o = x.shape[0] - R + 1
    cols = np.empty((B, C, R, R, ))
    for h in range(R):
        for w in range(R):
            cols[:,:, h, w, :, :] = x[:, :, h : h + o, w : w + o]
    #toDo...
    result = cols.transopse(1,2,3,0,4,5).reshape()


X, Y = unpickle("./3D/cifar-10-batches-py/data_batch_1")
filter = np.array([0, -0.2, 0, -0.2, 1.8, -0.2, 0, -0.2, 0]).reshape(3, 3)
print(convolution_2d(np.arange(16).reshape(4, 4), filter))
