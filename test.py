import numpy as np
import mnist
import matplotlib.pyplot as plt
from pylab import cm
import sys

A = np.arange(10).reshape(2, 5)/10


print(A)
print(np.exp(-1*A))

