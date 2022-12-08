import numpy as np
import sys, os
sys.path.append(os.path.dirname(__file__))
import Adam

class conv2D():
    def __init__(self, K, R, imglen, B):
        #filterはK * R * R (ch = 1 の場合のみを考えている)
        self.K = K
        self.R = R
        self.imr = imglen
        self.B = B
        self.filter_W = np.random.normal(loc=0, scale=0.01, size=(K, R * R))
        self.bias = np.repeat(np.random.normal(loc=0, scale=0.01, size=(K, 1)), imglen*imglen*B, axis=1)
        #
        self.filAdam = Adam.Adam(self.filter_W)
        self.biAdam = Adam.Adam(self.bias)
    
    #使う予定なし
    def fil2W(self, filter):
        K, self.R, _ = filter.shape
        return filter.reshape(K, self.R*self.R)

    def x2X(self, x, R):
        # x = B * 28 * 28を想定
        B, x_length, x_width = x.shape
        dx = x_length - R + 1
        dy = x_width - R + 1
        altx = np.zeros((B, R, R, dx, dy))
        for i in range(R):
            for j in range(R):
                altx[:, i, j, :, :] = x[:, i:i+dx, j:j+dy]
        return altx.transpose(1, 2, 0, 3, 4).reshape(R*R, dx*dy*B)
    
    def conv2D(self, x):
        #filter = K * R * Rを想定, Rは奇数を想定
        #x = B * x * x (x = 28)
        # 教科書通りに定義
        r = self.R // 2
        x_prime = np.pad(x, [(0,), (r,), (r,)], "constant")
        self.X = self.x2X(x_prime, self.R)
        self.Y = np.dot(self.filter_W, self.X) + self.bias
        # self.Y -> K * (B * imgsize)
        self.Y = self.Y.reshape(self.K, self.B, self.imr, self.imr).transpose(1, 0, 2, 3)
        return self.Y

    def back(self, delta):
        delta_filter_x = np.matmul(self.filter_W.T, delta)
        delta_filter_W = np.matmul(delta, self.X.T)
        delta_filter_b = np.sum(delta, axis=0)
        self.filter_W = self.filAdam.update(delta_filter_W)
        self.bias = self.biAdam.update(delta_filter_b)
        return delta_filter_x

    def test(self, x, filter_W, bias):
        r = self.R // 2
        x_prime = np.pad(x, [(0,), (r,), (r,)], "constant")
        X = self.x2X(x_prime, self.R)
        Y = np.dot(filter_W, X) + bias
        return Y.reshape(self.K, self.B, self.imr, self.imr)
