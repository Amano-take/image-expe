import numpy as np

class Dropout():
    def __init__(self, phi, M, B):
        self.phi = phi
        self.msk_num = int(M*phi)
        self.B = B

        drop_random = np.random.randint(0, M, self.msk_num*B).reshape(B, self.msk_num)
        mask_vector = np.ones((B, M))
        mask_vector[np.repeat(np.arange(B), self.msk_num), drop_random.flatten()] = 0
        self.msk = mask_vector.reshape(B, -1, 1)

    def prop(self, x):
        return x * self.msk

    def back(self, delta):
        return delta * self.msk.reshape(self.B, -1)