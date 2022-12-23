import numpy as np

class Dropout():
    def __init__(self, phi, B, M):
        self.phi = phi
        self.msk_num = int(M*phi)
        self.B = B
        self.M = M
        
    
    def prop(self, x):
        B = x.shape[0]
        M = self.M
        drop_random = np.repeat(np.random.randint(0, M, self.msk_num), B).reshape(B, -1)
        mask_vector = np.ones((B, M))
        mask_vector[np.repeat(np.arange(B), self.msk_num), drop_random.flatten()] = 0
        self.msk = mask_vector.reshape(B, -1, 1)
        return x * self.msk

    def back(self, delta):
        return delta * self.msk.reshape(self.B, self.M).transpose(1, 0)

    def test(self, x):
        return x * (1 - self.phi)