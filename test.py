import numpy as np
import mnist
import matplotlib.pyplot as plt
from pylab import cm
import sys


x = np.arange(18).reshape(2, 3, 3)

b_filter = np.repeat(np.random.normal(loc=0, scale=0.01, size=(3, 1)), 3, axis=1)


print(b_filter)

