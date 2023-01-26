import numpy as np
import mnist
import matplotlib.pyplot as plt
from pylab import cm
import sys

list_arr = []
with open("./contest/le4MNIST_X.txt") as f:
        for line in f:
            line = line.rstrip()
            l = line.split()
            arr = list(map(int, l))
            list_arr.append(arr)

Xtest = np.array(list_arr)
answer = []
for i in range(50):
    plt.imshow(Xtest[ 4*i:4*i + 4].reshape(2, 2,28,28).transpose(0, 2, 1, 3).reshape(56, 56), cmap=cm.gray)
    plt.show()
    string_idx = input().split(",")
    idx = list(map(int, string_idx))
    answer.extend(idx)

np.savetxt("./contest/answer2.txt", answer, fmt="%.0f")