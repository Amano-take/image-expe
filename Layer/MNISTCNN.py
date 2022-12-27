import mnist
import numpy as np
import sys, os
from decimal import Decimal, ROUND_HALF_UP
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import matplotlib.pyplot as plt
from Layer import BatchNormalize
from Layer import Pooling
from Layer import Conv3D
from Layer import Initialize
from Layer import Softmax_cross
from Layer import Activate
from Layer import Affine
from Layer import Diversify
from Layer.ConAn import ConAn
from Layer import Dropout
from Layer.Imshow import Imshow
from Layer.RandomArgumentation import RArg

class mCNN():
    def __init__(X, Y) -> None:
        B = 100
        C = 10
        Ini = Initialize.Initialize(X, Y, B, C, 0)

    def prop(self, X, Y):
        pass

