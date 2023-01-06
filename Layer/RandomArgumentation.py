import numpy as np
import mnist
import matplotlib.pyplot as plt
from pylab import cm
import Imshow
class RArg():

    def __init__(self, gauss, cutlen, theta, exrate, mleng, squetheta, wlen):
        #ガウシアンノイズ
        self.gauss = gauss
        #cutout
        self.cutlen = cutlen
        #回転角
        self.theta = theta
        #拡大率
        self.exrate = exrate
        #平行移動幅
        self.mleng = mleng
        #スキュー角
        self.squetheta = squetheta
        #ホワイトノイズ
        self.wlen = wlen
    
    def prop2(self, img, x):
        #気付くのに時間がかかりすぎた...
        #配列はアドレスわたし
        image = np.copy(img)
        dice = np.random.uniform(0, 1, (3, ))
        p_affin =  6 * x / ( 7 * (20 + x))
        p_musk =  x / ( 10 *  (20 + x))
        p_noise = 1 * x / (10 * (20 + x))
        k_affin =  x /(2 * (200 + x)) + 1/2
        k_musk =    x /( (50 + x))
        k_noise =   x /(2 * (50 + x)) + 1/2
        if(dice[0] < 1/2):
            theta = np.random.uniform(-self.theta, self.theta) * k_affin
            af1 = np.array([[np.cos(theta),-np.sin(theta)],
                        [np.sin(theta),np.cos(theta)]])
            xrate, yrate = np.random.uniform(1 / self.exrate , self.exrate, (2,)) 
            xrate = (1 - xrate) * k_affin + xrate
            yrate = (1 - yrate) * k_affin + yrate
            af2 = np.array([xrate, 0, 0, yrate]).reshape(2,2)
            xsque, ysque = np.random.uniform(-self.squetheta, self.squetheta, (2,)) * k_affin
            af3 = np.array([1, np.tan(xsque), np.tan(ysque), 1]).reshape(2,2)
            af = af3 @ af2 @ af1
            image = self.Affine_conv(image, af)
            x, y = np.random.randint(-self.mleng, self.mleng, (2, )) * k_affin
            x = x.astype(int)
            y = y.astype(int)
            image = self.translation(x, y, image)
        if(dice[2] < 1/10):
            image = self.thick_filtering(image)
            image = self.whitenoise(image, self.wlen)
        if(dice[1] < 1 ):
            image = self.addnoise(image, k_musk)
            """muskdice = np.random.randint(0, 2)
            if(muskdice == 0):
                cutlen = int(self.cutlen * k_musk)
                image = self.cutout(image, cutlen)
            else:
                image = self.randomerase(image, k_musk)"""
        return image
        
    def prop(self, image):
        #平行移動, 回転, 拡大縮小, cutout, ガウシアンノイズ, スキューが1/5ずつで選ばれる
        #組み合わせが表現できないのでは？？全て1/choiceに変更
        dice = np.random.randint(0, 6, (7,))
        if(dice[0] < 3):
            x, y = np.random.randint(-self.mleng, self.mleng, (2,))
            image = self.translation(x, y, image)
        if(dice[1] < 3):
            theta = np.random.uniform(-self.theta, self.theta)
            af = np.array([[np.cos(theta),-np.sin(theta)],
                        [np.sin(theta),np.cos(theta)]])
            image = self.Affine_conv(image, af) 
        if(dice[2]  < 2):
            xrate, yrate = np.random.uniform(1, self.exrate, (2,))
            yrate = yrate * yrate
            af = np.array([xrate, 0, 0, yrate]).reshape(2,2)
            image = self.Affine_conv(image, af)
        if(dice[3] == 0):
        #太くする
            image = self.thick_filtering(image)
        if(dice[4] < 1):
            image = self.addnoise(image)
        if(dice[5] < 1):
            xsque, ysque = np.random.uniform(-self.squetheta, self.squetheta, (2,))
            af = np.array([1, np.tan(xsque), np.tan(ysque), 1]).reshape(2,2)
            image = self.Affine_conv(image, af)
        if(dice[6] < 3):
            image = self.cutout(image, self.cutlen)
        #elif(dice[6] > 5):
            #image = self.whitenoise(image, self.wlen)
        #elif(dice[6] == 4):
            #image = self.randomcrop(image)
        return image

    """def prop(self, image):
        x, y = np.round(np.random.normal(loc=0, scale=self.mleng, size=(2,))).astype(np.int)
        image = self.translation(x, y, image)
        theta = np.random.normal(loc=0, scale=self.theta)
        af = np.array([[np.cos(theta),-np.sin(theta)],
                        [np.sin(theta),np.cos(theta)]])
        image = self.Affine_conv(image, af)
        xrate, yrate = np.random.normal(loc=1, scale=self.exrate, size=(2,))
        af = np.array([xrate, 0, 0, yrate]).reshape(2,2)
        image = self.Affine_conv(image, af)
        dice = np.random.uniform(0, 6, (2,))
        if(dice[0] < 2):
            image = self.thick_filterling(image)
        image = self.addnoise(image)
        xsque, ysque = np.random.normal(0, self.squetheta, (2,))
        af = np.array([1, np.tan(xsque), np.tan(ysque), 1]).reshape(2,2)
        image = self.Affine_conv(image, af)
        if(dice[1] < 1.5):
            image = self.cutout(image, self.cutlen)
        elif(dice[1] < 3):
            image = self.whitenoise(image, self.wlen)
        elif(dice[1] < 4.5):
            image = self.randomcrop(image)
        return image"""

    def translation(self, x, y, img):
        z = np.abs(x) + np.abs(y)
        B, imlen, _ = img.shape
        big_img = np.zeros((B, imlen+2*(z), imlen+2*(z)))
        big_img[:, z:imlen+z, z:imlen+z] = img
        ans_img = np.zeros((B, imlen, imlen))
        ans_img[:, :, :] = big_img[:, z - x:z - x+imlen, z - y:z - y+imlen]
        return ans_img

    def Affine_conv(self, image, affin):
        len = image.shape[1]
        x,y = np.mgrid[:len,:len]
        #格子点を画素
        xy_after = np.dstack((x,y))

        #affine変換行列

        inv_affin = np.linalg.inv(affin)
        #画像を中心に持っていく
        xy_after_o = xy_after - len//2
        #変換後に軸に垂直になるような元の座標を取得
        #ref_xy = np.einsum('ijk,lk->ijl',xy_after_o,inv_affin)[...,:2]
        ref_xy_o = np.dot(inv_affin,xy_after_o.reshape(len*len,2,1)).transpose(1,0,2).reshape(-1, 2)
        #画像を元の位置に　ref_xyは回転前の座標である
        ref_xy = ref_xy_o + len//2 + len//2

        #左下をintで計算した後、それを移動させて計算
        linear_xy = {}
        linear_xy['upleft'] = ref_xy.astype(int) - len//2
        linear_xy['downleft'] = linear_xy['upleft'] + [1,0]
        linear_xy['upright']= linear_xy['upleft'] + [0,1]
        linear_xy['downright'] = linear_xy['upleft'] + [1,1]
        #linear_xy[key] = [[格子点0'], [格子点1'].....]
        ref_xy -= len//2
        upleft_diff = ref_xy - linear_xy['upleft']
        #(1-xの差)と(1-yの差)の積を計算
        linear_weight = {}
        linear_weight['upleft'] = (1-upleft_diff[:,0])*(1-upleft_diff[:,1])
        linear_weight['downleft'] = upleft_diff[:,0]*(1-upleft_diff[:,1])
        linear_weight['upright'] = (1-upleft_diff[:,0])*upleft_diff[:,1]
        linear_weight['downright'] = upleft_diff[:,0]*upleft_diff[:,1]
        # imgを拡大(傾ける前の画像をpddingすることによって、参照できないということがないように)
        #img_big = np.pad(img, [0, (28,), (28,)], 'constant')
        img_big = np.zeros([image.shape[0],len*5,len*5])
        img_big[:,len * 2:len*3,len*2:len*3] = image

        linear_with_weight = {}
        for direction in linear_xy.keys():
            xy = linear_xy[direction]
            weight = linear_weight[direction]
            
            linear_with_weight[direction] = np.einsum('i,ki->ki',weight,img_big[:,xy[...,0]+len*2,xy[...,1]+len*2])

        img_linear = sum(linear_with_weight.values()).reshape(image.shape[0], len,len)
        return img_linear

    def cutout(self, img, cutlen):
        B, len, _ = img.shape
        big_img = np.zeros((B, len*3, len*3))
        big_img[:, len:len*2, len:len*2] = img
        a, b = np.random.randint(0, 28, (2,))
        x = cutlen // 2
        y = cutlen // 2
        big_img[:, a+len-x:a+len+x, b+len-y:b+len+y] = np.average(img)
        return big_img[:, len:len*2, len:len*2]
    
    def whitenoise(self, img, wlen):
        B, len, _ = img.shape
        c = np.max(img)
        big_img = np.zeros((B, len*3, len*3))
        big_img[:, len:len*2, len:len*2] = img
        a, b = np.random.randint(0, 28, (2,))
        x, y = np.random.randint(0, wlen//2 + 1, (2,))
        big_img[:, a+len-x:a+len+x, b+len-y:b+len+y] = np.random.randint(0, c, (2*x, 2*y))
        return big_img[:, len:len*2, len:len*2]

    def randomerase(self, img, k):
        B, len, _ = img.shape
        se = np.random.uniform(0.02, 0.4) * 28 * 28 * k
        re = np.random.uniform(3/10, 10/3)
        x = np.sqrt(se/re).astype(int)
        y = np.sqrt(se*re).astype(int)
        xe, ye = np.random.randint(0, len, (2,))
        if (xe + x >= len or ye + y >= len):
            self.randomerase(img, k)
        else:
            img[:, xe:xe+x, ye:ye+y] = np.random.randint(0, 255, (x, y))
        
        return img
    
    def addnoise(self, img, k):
        B, len, _ = img.shape
        noise = np.random.normal(loc=1, scale=(k * self.gauss), size=(B, len, len))
        return img * noise
    
    def randomcrop(self, img):
        B, len, _ = img.shape
        cropimage = np.zeros((B, len, len))
        a, b = np.random.randint(10, 18, (2,))
        len, wid = np.random.randint(7, 10, (2,))
        cropimage[:, a-len:a+len, b-wid:b+wid] = img[:, a-len:a+len, b-wid:b+wid]
        return cropimage
    

    def x2X(self, x, R):
        # x = B * ch * 28+r * 28+rを想定
        B, x_length, x_width = x.shape
        dx = x_length - R + 1
        dy = x_width - R + 1
        altx = np.zeros((B, R, R, dx, dy))
        for i in range(R):
            for j in range(R):
                altx[:, i, j, :, :] = x[:, i:i+dx, j:j+dy]
        return altx.transpose(1, 2, 0, 3, 4).reshape(R*R, dx*dy*B)
    
    def thick_filtering(self, x):
        #filter_w = K * (ch * R * R), Rは奇数を想定
        #x = B * ch * x * x (x = 28)
        # 教科書通りに定義
        B, imr, _ = x.shape
        R = 3
        K = 1
        filter_W = np.array([0.5, 0.5, 0.5, 0.5, 5,0.5, 0.5, 0.5,0.5]).reshape(1, 9)
        bias = 0
        r = R // 2
        self.r = r
        x_prime = np.pad(x, [(0, ), (r,), (r,)], "constant")
        X = self.x2X(x_prime, R)
        Y = np.dot(filter_W, X) + bias
        Y = Y.reshape(K, B, imr, imr).reshape(B, imr, imr)
        return np.where(Y>=255, 255, Y)

    def thin_filtering(self, x):
        B, imr, _ = x.shape
        R = 3
        K = 1
        filter_W = np.array([0, 1/4, 0, 1/4, 0,1/4,0,1/4,0]).reshape(1, 9)
        bias = 0
        r = R // 2
        self.r = r
        x_prime = np.pad(x, [(0, ), (r,), (r,)], "constant")
        X = self.x2X(x_prime, R)
        Y = np.dot(filter_W, X) + bias
        Y = Y.reshape(K, B, imr, imr).reshape(B, imr, imr)
        Y = np.where(Y<=150, 0, Y)
        return np.where(Y>=255, 255, Y)


#
"""
X = np.array(mnist.download_and_parse_mnist_file("train-images-idx3-ubyte.gz"))
img = X[0:100].reshape(100, 28, 28)
ims = Imshow.Imshow()
#plt.imshow(img[0], cmap = cm.gray)
#plt.show()

#sque角がでかいとエラー確認
ra = RArg(0.05, 10, np.pi/5, 5/4, 4, np.pi/9, 5)
index = 65
shimg = np.copy(img[index])
for i in range(100):
    cimg = ra.prop2(img, i)
    shimg = np.vstack((shimg, cimg[index]))
        
shimg = shimg.reshape(-1, 28, 28)
ims.imshow(shimg)
#ims.imshow(ra.prop(img))
#"""