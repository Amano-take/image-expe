import numpy as np
import Affine

class test():
    def __init__(self):
        M = 80
        img_size = 100
        W1 = np.random.normal(loc=0, scale=np.sqrt(
            1/img_size), size=(M, img_size))
        b1 = np.random.normal(loc=0, scale=np.sqrt(1/img_size), size=(M, 1))
        Layer = Affine.Affine(W1, b1)
        x = np.ones(300).reshape(3, 100, 1)
        y = Layer.prop(x)
        x_prime = Layer.back(np.ones(240).reshape(80, 3))
        print(Layer.W.shape)
test()