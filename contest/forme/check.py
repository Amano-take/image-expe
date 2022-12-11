import numpy as np
import matplotlib.pyplot as plt
from pylab import cm
import mnist
import sys
import os
sys.path.append(os.path.join(os.path.join(os.path.dirname(__file__), ".."), ".."))

list_arr = []
with open("./contest/le4MNIST_X.txt") as f:
    for line in f:
        line = line.rstrip()
        l = line.split()
        arr = list(map(int, l))
        list_arr.append(arr)

Y = np.loadtxt("./contest/predict.txt")

X = np.array(list_arr).reshape(-1, 28, 28)
for i in range(10):
    print("0 ~ 9999")
    index = int(input())
    print(Y[index:index+12].reshape(3, 4))
    plt.imshow(X[index:index+12].reshape(3, 4, 28, 28).transpose(0,
                                                                 2, 1, 3).reshape(3*28, 4*28), cmap=cm.gray)
    plt.show()
    
