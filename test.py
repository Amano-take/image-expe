import numpy as np
import mnist
import matplotlib.pyplot as plt
from pylab import cm
import sys

X = np.arange(20).reshape(2, 2, 5)
Y = np.arange(30).reshape(3,2,5)
print(np.vstack((X, Y)))

