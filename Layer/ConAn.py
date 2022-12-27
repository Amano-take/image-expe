import numpy as np
import mnist

class ConAn():

    def __init__(self):
        pass

    @staticmethod
    def get(x, y):
        list_arr = []
        with open("./contest/le4MNIST_X.txt") as f:
            for line in f:
                line = line.rstrip()
                l = line.split()
                arr = list(map(int, l))
                list_arr.append(arr)
        Xtest = np.array(list_arr).reshape(-1, 28, 28)[x:y]
        list_ans = []
        with open("./contest/forme/answer2.txt") as f:
            for line in f:
                line = line.rstrip()
                arr = list(map(int, line))
                list_ans.append(arr)
        Ytest = np.array(list_ans).flatten()[x:y]

        return Xtest, Ytest

    @staticmethod
    def gettest():
        list_arr = []
        with open("./contest/le4MNIST_X.txt") as f:
            for line in f:
                line = line.rstrip()
                l = line.split()
                arr = list(map(int, l))
                list_arr.append(arr)
        Xtest = np.array(list_arr).reshape(-1, 28, 28)
        return Xtest
    
    @staticmethod
    def getmnist(arg=70000):
        X_p = np.array(mnist.download_and_parse_mnist_file(
            "train-images-idx3-ubyte.gz"))
        Y_p = np.array(mnist.download_and_parse_mnist_file(
            "train-labels-idx1-ubyte.gz"))
        Xtest = np.array(mnist.download_and_parse_mnist_file(
            "t10k-images-idx3-ubyte.gz"))
        Ytest = np.array(mnist.download_and_parse_mnist_file(
            "t10k-labels-idx1-ubyte.gz"))
        X = np.vstack((X_p, Xtest))
        Y = np.hstack((Y_p, Ytest))
        return X[0:arg], Y[0:arg]

    @staticmethod
    def onehot(answer):
        onehot = np.zeros((answer.size, 10))
        onehot[np.arange(answer.size), answer] = 1
        onehot = onehot.reshape(-1, 10, 1)
        return onehot