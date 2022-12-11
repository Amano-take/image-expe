import numpy as np

x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]).reshape(3,2,2)
y = np.array(np.arange(8)).reshape(2,2,2)
print(x)
print(np.vstack((x, y)).shape)