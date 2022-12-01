import numpy as np
import mnist
import matplotlib.pyplot as plt
from pylab import cm
import sys


x = np.array([1,2,12,34,23,12,3532,123,32,4, 12, 9]).reshape(3, 4, 1)
print(np.sum(x, axis = 0).shape)

print(x)
microB = np.sum(x, axis = 0) / x.shape[0]
sigmaB = np.sum((x - microB) ** 2, axis = 0) / x.shape[0]
normalize_x = (x - microB) / np.sqrt(sigmaB + 1e-12)
print(sigmaB)
y = normalize_x 
print(y)




