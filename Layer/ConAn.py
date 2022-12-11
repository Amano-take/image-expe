import numpy as np

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