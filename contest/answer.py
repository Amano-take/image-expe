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
for i in range(100):
    plt.imshow(Xtest[i].reshape(28,28), cmap=cm.gray)
    plt.show()
    string_idx = input()
    idx = int(string_idx)
    answer.append(idx)

np.savetxt("./contest/answer.txt", answer, fmt="%.0f")