import numpy as np
import matplotlib.pyplot as plt
from pylab import cm

class Imshow():
    def __init__(self) -> None:
        pass

    @staticmethod
    def imshow(X):
        for i in range(10):
            #print(Y[index:index+12].reshape(3, 4))
            plt.imshow(X[i*12:i*12+12].reshape(3, 4, 28, 28).transpose(0,
                                                                        2, 1, 3).reshape(3*28, 4*28), cmap=cm.gray)
            plt.show()

    @staticmethod
    def imansshow(X,Y):
        for i in range(10):
            print(Y[i*12:i*12+12])
            plt.imshow(X[i*12:i*12+12].reshape(3, 4, 28, 28).transpose(0,
                                                                        2, 1, 3).reshape(3*28, 4*28), cmap=cm.gray)
            plt.show()