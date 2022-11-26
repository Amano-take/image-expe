import numpy as np
import mnist
import matplotlib.pyplot as plt
from pylab import cm
import sys


A = np.ones(10).reshape(2, 5)

B=np.array([[1,2,3],[3,1,4]])

A[np.repeat(np.arange(2), 3), B.flatten()] = 0

C = np.ones(10).reshape(2,5) + 1

C = np.where(A == 0, 0, C)

print(A)




