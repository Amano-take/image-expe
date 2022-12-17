import numpy as np

class MSGD():
    eta = 0.01
    alpha = 0.9
    def __init__(self, para):
        #更新するパラメータ
        self.para = para
        self.deltaW = 0

    def update(self, delta):
        self.deltaW = MSGD.alpha * self.deltaW - MSGD.eta * delta
        self.para = self.para + self.deltaW
        return self.para
        

class ADSGD():
    alpha = 0.001
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-12
    ti = 10
    eta = 0.01
    def __init__(self, para):
        #Adam学習用
        self.m = 0
        self.v = 0
        self.t = 0
        #更新するパラメータ
        self.para = para
        eta = 0.01

    def update(self, delta, k):
        if(k >= ADSGD.ti):
            self.t = self.t + 1
            self.m = ADSGD.beta1 * self.m + (1 - ADSGD.beta1) * delta
            self.v = ADSGD.beta2 * self.v + (1 - ADSGD.beta2) * delta * delta
            m_head = self.m / (1 - np.power(ADSGD.beta1, self.t))
            v_head = self.v / (1 - np.power(ADSGD.beta2, self.t))
            self.para = self.para - ADSGD.alpha * m_head / \
                (np.power(v_head, 1/2) + ADSGD.eps)
            return self.para
        else:
            self.para = self.para - ADSGD.eta * delta
            return self.para