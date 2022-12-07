import numpy as np

class Adam():
    #Adamパラメータ
    alpha = 0.001
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    def __init__(self, para):
        #Adam学習用
        self.m = 0
        self.v = 0
        self.t = 0
        #更新するパラメータ
        self.para = para

    def update(self, delta):
        self.t = self.t + 1
        self.m = Adam.beta1 * self.m + (1 - Adam.beta1) * delta
        self.v = Adam.beta2 * self.v + (1 - Adam.beta2) * delta * delta
        m_head = self.m / (1 - np.power(Adam.beta1, self.t))
        v_head = self.v / (1 - np.power(Adam.beta2, self.t))
        self.para = self.para - Adam.alpha * m_head / \
            (np.power(v_head, 1/2) + Adam.eps)
        return self.para
        