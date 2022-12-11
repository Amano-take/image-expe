import numpy as np
import mnist
import matplotlib.pyplot as plt
from pylab import cm
import sys

X = np.array(mnist.download_and_parse_mnist_file("train-images-idx3-ubyte.gz"))
img = X[0].reshape(1, 28, 28)

B, len, _ = img.shape
noise = np.random.normal(loc=1, scale=0.1, size=(B, len, len))
plt.imshow((img*noise)[0], cmap = cm.gray)
plt.show()

