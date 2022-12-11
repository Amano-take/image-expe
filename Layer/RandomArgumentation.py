import numpy as np
import mnist
import matplotlib.pyplot as plt
from pylab import cm

class RArg():

    def __init__(self, gauss, cutlen, theta, exrate, shrate, mleng, squetheta, wlen):
        #ガウシアンノイズ
        self.gauss = gauss
        #cutout
        self.cutlen = cutlen
        #回転角
        self.theta = theta
        #拡大率
        self.exrate = exrate
        #縮小率
        self.shrate = shrate
        #平行移動幅
        self.mleng = mleng
        #スキュー角
        self.squetheta = squetheta
        #ホワイトノイズ
        self.wlen = wlen

    def prop(self, image):
        #平行移動, 回転, 拡大縮小, cutout, ガウシアンノイズ, スキューが1/5ずつで選ばれる
        #組み合わせが表現できないのでは？？全て1/choiceに変更
        choice = 6
        dice = np.random.randint(0, choice, (choice,))
        if(dice[0] == 0):
            x, y = np.random.randint(-self.mleng, self.mleng, (2,))
            image = self.translation(x, y, image)
        if(dice[1] < 3):
            theta = np.random.uniform(-self.theta, self.theta)
            af = np.array([[np.cos(theta),-np.sin(theta)],
                        [np.sin(theta),np.cos(theta)]])
            image = self.Affine_conv(image, af) 
        if(dice[2] == 0):
            xrate, yrate = np.random.uniform(self.shrate, self.exrate, (2,))
            af = np.array([xrate, 0, 0, yrate]).reshape(2,2)
            image = self.Affine_conv(image, af)
        
        if(dice[4] < 3):
            image = self.addnoise(image)
        if(dice[5] < 2):
            xsque, ysque = np.random.uniform(-self.squetheta, self.squetheta, (2,))
            af = np.array([1, np.tan(xsque), np.tan(ysque), 1]).reshape(2,2)
            image = self.Affine_conv(image, af)
        if(dice[3] < 4):
            image = self.cutout(image, self.cutlen)
        elif(dice[3] > 4):
            image = self.whitenoise(image, self.wlen)
        elif(dice[3] == 4):
            image = self.randomcrop(image)

        
        return image

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
        big_img[:, a+len:a+len+cutlen, b+len:b+len+cutlen] = 0
        return big_img[:, len:len*2, len:len*2]
    
    def whitenoise(self, img, wlen):
        B, len, _ = img.shape
        c = np.max(img)
        big_img = np.zeros((B, len*3, len*3))
        big_img[:, len:len*2, len:len*2] = img
        a, b = np.random.randint(0, 28, (2,))
        x, y = np.random.randint(0, wlen, (2,))
        big_img[:, a+len:a+len+x, b+len:b+len+y] = c
        return big_img[:, len:len*2, len:len*2]

    def addnoise(self, img):
        B, len, _ = img.shape
        noise = np.random.normal(loc=1, scale=self.gauss, size=(B, len, len))
        return img * noise
    
    def randomcrop(self, img):
        B, len, _ = img.shape
        cropimage = np.zeros((B, len, len))
        a, b = np.random.randint(4, 12, (2,))
        len, wid = np.random.randint(5, 16, (2,))
        cropimage[:, a:a+len, b:b+wid] = img[:, a:a+len, b:b+wid]
        return cropimage

"""
X = np.array(mnist.download_and_parse_mnist_file("train-images-idx3-ubyte.gz"))
img = X[0].reshape(1, 28, 28)
ra = RArg(0.1, 7, np.pi/3, 5/4, 4/5, 7, np.pi/8, 4) #sque角がでかいとエラー確認
plt.imshow(ra.prop(img)[0], cmap =cm.gray)
plt.show()
#"""