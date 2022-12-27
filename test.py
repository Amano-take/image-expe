import numpy as np
import mnist
import matplotlib.pyplot as plt
from pylab import cm
import sys

x = np.empty((0, 5), dtype=int)
y = np.arange(10).reshape(2,5)

print(np.vstack((x, y, y)))