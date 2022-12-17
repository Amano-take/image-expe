import numpy as np
import Adam

class SAM():
    rho = 0.05
    phi = 0.01
    def __init__(self, para):
        self.para = para
        self.paAdam = Adam.Adam(para)

    def calw(self, delta):
        self.eps_w = SAM.rho * delta / np.linalg.norm(delta, ord=2)
        return self.para + self.eps_w

    def update(self, delta):
        self.para = self.para - self.eps_w - SAM.phi * delta
        return self.para
