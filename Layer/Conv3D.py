import numpy as np
import Adam
import SAM
import MomentumSGD

class Conv3D():

    def __init__(self, K, R, imglen, B, ch, l2para=0, opt="Adam"):
        #filterはK * ch * R * R
        #入力はB *　ch * imglen * imglenを想定
        self.l2lambda = l2para
        self.K = K
        self.R = R
        self.imr = imglen
        self.B = B
        self.ch = ch
        self.opt = opt
        #フィルターの初期化
        self.filter_W = np.random.normal(loc=1/(R*R), scale=0.01, size=(K, R * R * ch))
        self.bias = np.repeat(np.random.normal(loc=0, scale=0.01, size=(K, 1)), imglen*imglen*B, axis=1)
        #
        if(opt == "Adam"):
            self.filAdam = Adam.Adam(self.filter_W)
            self.biAdam = Adam.Adam(self.bias)
        elif(opt == "MSGD"):
            self.filAdam = MomentumSGD.MSGD(self.filter_W)
            self.biAdam = MomentumSGD.MSGD(self.bias)
        elif(opt == "SAM"):
            self.filsam = SAM.SAM(self.filter_W)
            self.bsam = SAM.SAM(self.bias)
        elif(opt=="ADSGD"):
            self.filAdam = MomentumSGD.ADSGD(self.filter_W)
            self.biAdam = MomentumSGD.ADSGD(self.bias)
    
    #使う予定なし
    def fil2W(self, filter):
        K, ch, self.R, _ = filter.shape
        return filter.reshape(K, -1)

    def x2X(self, x, R):
        # x = B * ch * 28+r * 28+rを想定
        B, ch, x_length, x_width = x.shape
        dx = x_length - R + 1
        dy = x_width - R + 1
        altx = np.zeros((B, ch, R, R, dx, dy))
        for i in range(R):
            for j in range(R):
                altx[:, :, i, j, :, :] = x[:, :, i:i+dx, j:j+dy]
        return altx.transpose(1, 2, 3, 0, 4, 5).reshape(R*R*ch, dx*dy*B)
    
    def prop(self, x):
        #filter_w = K * (ch * R * R), Rは奇数を想定
        #x = B * ch * x * x (x = 28)
        # 教科書通りに定義
        r = self.R // 2
        #backで使用
        self.r = r
        x_prime = np.pad(x, [(0,), (0, ), (r,), (r,)], "constant")
        self.X = self.x2X(x_prime, self.R)
        self.Y = np.dot(self.filter_W, self.X) + self.bias
        # self.Y -> K * (imgsize * B)
        self.Y = self.Y.reshape(self.K, self.B, self.imr, self.imr).transpose(1, 0, 2, 3)
        # self.Y -> B * K * imlen * imlen
        return self.Y

    def back(self, delta):
        #delta -> B * K * imlen * imlen
        delta = delta.transpose(1, 0, 2, 3).reshape(self.K, -1)
        #delta -> K * (B * imlen * imlen)
        delta_filter_x = np.matmul(self.filter_W.T, delta)
        delta_filter_W = np.matmul(delta, self.X.T)
        delta_filter_b = np.sum(delta, axis=0)
        self.filter_W = self.filAdam.update(delta_filter_W)
        self.bias = self.biAdam.update(delta_filter_b)
        return self.X2x(delta_filter_x)[:, :, self.r:self.r + self.imr, self.r:self.r + self.imr]
    
    def X2x(self, X):
        w = self.R
        bigL = self.imr + self.r*2 - w + 1
        bigW = self.imr + self.r*2 - w + 1
        x = np.zeros((self.B, self.ch, self.imr + self.r * 2, self.imr + self.r * 2))
        arX = X.reshape(self.ch, w, w, self.B, bigL, bigW).transpose(3, 0, 1, 2, 4, 5)
        for i in range(w):
            for j in range(w):
                x[:, :, i:i+bigL, j:j+bigW] = arX[:, :, i, j, :, :]
        return x

    def test(self, x, filter_W, bias):
        r = self.R // 2
        x_prime = np.pad(x, [(0,), (0,), (r,), (r,)], "constant")
        X = self.x2X(x_prime, self.R)
        Y = np.dot(filter_W, X) + bias
        return Y.reshape(self.K, self.B, self.imr, self.imr).transpose(1, 0, 2, 3)

    def update(self, filter_W, bias):
        self.filter_W = filter_W
        self.bias = bias
        opt = self.opt
        if(opt == "Adam"):
            self.filAdam = Adam.Adam(self.filter_W)
            self.biAdam = Adam.Adam(self.bias)
        elif(opt == "MSGD"):
            self.filAdam = MomentumSGD.MSGD(self.filter_W)
            self.biAdam = MomentumSGD.MSGD(self.bias)
        elif(opt == "SAM"):
            self.filsam = SAM.SAM(self.filter_W)
            self.bsam = SAM.SAM(self.bias)
        elif(opt=="ADSGD"):
            self.filAdam = MomentumSGD.ADSGD(self.filter_W)
            self.biAdam = MomentumSGD.ADSGD(self.bias)

    def SAM_back1(self, delta):
        delta = delta.transpose(1, 0, 2, 3).reshape(self.K, -1)
        #delta -> K * (B * imlen * imlen)
        delta_filter_x = np.matmul(self.filter_W.T, delta)
        delta_filter_W = np.matmul(delta, self.X.T)
        delta_filter_b = np.sum(delta, axis=0)
        self.filter_W = self.filsam.calw(delta_filter_W)
        self.bias = self.bsam.calw(delta_filter_b)
        return self.X2x(delta_filter_x)[:, :, self.r:self.r + self.imr, self.r:self.r + self.imr]

    def SAM_back2(self, delta):
        delta = delta.transpose(1, 0, 2, 3).reshape(self.K, -1)
        #delta -> K * (B * imlen * imlen)
        delta_filter_x = np.matmul(self.filter_W.T, delta)
        delta_filter_W = np.matmul(delta, self.X.T)
        delta_filter_b = np.sum(delta, axis=0)
        self.filter_W = self.filsam.update(delta_filter_W)
        self.bias = self.bsam.update(delta_filter_b)
        return self.X2x(delta_filter_x)[:, :, self.r:self.r + self.imr, self.r:self.r + self.imr]
    
    def back_ADSGD(self, delta, k):
        #delta -> B * K * imlen * imlen
        delta = delta.transpose(1, 0, 2, 3).reshape(self.K, -1)
        #delta -> K * (B * imlen * imlen)
        delta_filter_x = np.matmul(self.filter_W.T, delta)
        delta_filter_W = np.matmul(delta, self.X.T)
        delta_filter_b = np.sum(delta, axis=0)
        self.filter_W = self.filAdam.update(delta_filter_W, k)
        self.bias = self.biAdam.update(delta_filter_b, k)
        return self.X2x(delta_filter_x)[:, :, self.r:self.r + self.imr, self.r:self.r + self.imr]

    def back_l2(self, delta):
        #delta -> B * K * imlen * imlen
        delta = delta.transpose(1, 0, 2, 3).reshape(self.K, -1)
        #delta -> K * (B * imlen * imlen)
        delta_filter_x = np.matmul(self.filter_W.T, delta) 
        delta_filter_W = np.matmul(delta, self.X.T) + 2 * self.l2lambda * self.filter_W
        delta_filter_b = np.sum(delta, axis=0)+ 2 * self.l2lambda * self.bias
        self.filter_W = self.filAdam.update(delta_filter_W)
        self.bias = self.biAdam.update(delta_filter_b)
        return self.X2x(delta_filter_x)[:, :, self.r:self.r + self.imr, self.r:self.r + self.imr]
        
'''
con = Conv3D(3, 3, 4, 1, 2)
x = np.arange(32).reshape(1, 2, 4, 4)
con.prop(x)
delta = np.ones(48).reshape(1, 3, 4, 4)
print(con.back(delta).shape)
'''