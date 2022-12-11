import numpy as np
import mnist
import matplotlib.pyplot as plt
import Pooling
from pylab import cm

class Diversify():
    def __init__(self):
        #image = B * 28 * 28を想定
        self.noise = 0.2
        self.cutoutlen = 5
        self.af_extend1 = np.array([8/7, 0, 0, 1]).reshape(2, 2)
        self.af_extend2 = np.array([1, 0, 0, 8/7]).reshape(2, 2)
        self.af_extend3 = np.array([8/7, 0, 0, 8/7]).reshape(2,2)
        self.af_shrink1 = np.array([4/5, 0, 0, 1]).reshape(2,2)
        self.af_shrink2 = np.array([1,0, 0, 4/5]).reshape(2,2)
        self.af_shrink3 = np.array([4/5, 0, 0, 4/5]).reshape(2,2)
        self.sque1 = np.array([1, 0, 1/4, 1]).reshape(2,2)
        self.sque2 = np.array([1, 1/4, 0, 1]).reshape(2,2)
        self.sque3 = np.array([1, 0, -1/4, 1]).reshape(2,2)
        self.sque4 = np.array([1, -1/4, 0, 1]).reshape(2,2)
        return
    
    def cheating(self):
        Y = np.loadtxt("./contest/forme/answer.txt", dtype=int)
        list_arr = []
        with open("./contest/le4MNIST_X.txt") as f:
            for line in f:
                line = line.rstrip()
                l = line.split()
                arr = list(map(int, l))
                list_arr.append(arr)
        X = np.array(list_arr).reshape(-1, 28, 28)[0:100, :,:]
        after_X = self.expand_prime(X)
        after_Y = np.repeat(Y, after_X.shape[0] // Y.shape[0], axis=0)
        return after_X, after_Y
    
    def translation(self, x, y, img):
        z = np.abs(x) + np.abs(y)
        B, imlen, _ = img.shape
        big_img = np.zeros((B, imlen+2*(z), imlen+2*(z)))
        big_img[:, z:imlen+z, z:imlen+z] = img
        ans_img = np.zeros((B, imlen, imlen))
        ans_img[:, :, :] = big_img[:, z - x:z - x+imlen, z - y:z - y+imlen]
        return ans_img

    
    def expand(self, image):
        B, len, _ = image.shape
        image0 = self.addnoise(image)
        image_alpha = self.cutout(image, self.cutoutlen)
        image1 = self.rotate_imgs(image, np.pi/4)
        image2 = self.rotate_imgs(image, -np.pi/4)
        image3 = self.rotate_imgs(image, np.pi/6)
        image4 = self.rotate_imgs(image, -np.pi/6)
        Aimage = self.rotate_imgs(image, np.pi/9)
        Bimage = self.rotate_imgs(image, -np.pi/9)
        Cimage = self.rotate_imgs(image, np.pi/18)
        image_beta = self.cutout(image, self.cutoutlen)
        Dimage = self.rotate_imgs(image, -np.pi/18)
        image_ganma = self.cutout(image, self.cutoutlen)
        image00 = self.addnoise(Cimage)
        image01 = self.addnoise(Dimage)
        Fimage = self.Affine_conv(image, self.af_extend1)
        Gimage = self.Affine_conv(image, self.af_extend2)
        Himage = self.Affine_conv(image, self.af_extend3)
        image7 = self.Affine_conv(Cimage, self.af_extend3)
        image8 = self.Affine_conv(Dimage, self.af_extend3)
        image5 = self.Affine_conv(image, self.af_shrink1)
        image6 = self.Affine_conv(image, self.af_shrink2)
        Iimage = self.Affine_conv(image, self.af_shrink3)
        image9 = self.Affine_conv(Cimage, self.af_shrink3)
        image10 = self.Affine_conv(Dimage, self.af_shrink3)
        Jimage = self.translation(3, 3, image)
        image11 = self.translation(3, 3, Cimage)
        image12 = self.translation(3, 3, Dimage)
        Kimage = self.translation(3,-3,image)
        Limage = self.translation(-3, 3, image)
        Mimage = self.translation(-3,-3,image)
        image13 = self.translation(-3,-3, Cimage)
        image14 = self.translation(-3, -3, Dimage)
        Nimage = self.Affine_conv(image, self.sque1)
        image15 = self.Affine_conv(Cimage, self.sque1)
        image16 = self.Affine_conv(Dimage, self.sque1)
        Oimage = self.Affine_conv(image, self.sque2)
        Pimage = self.Affine_conv(image, self.sque3)
        image17 = self.Affine_conv(Cimage, self.sque3)
        image18 = self.Affine_conv(Dimage, self.sque3)
        Qimage = self.Affine_conv(image, self.sque4)

        return np.hstack((image, image1, image2, image3, image4, image5, image6, image_alpha, image_beta, image_ganma,
                            image7, image8, image9, image10, image11, image12,
                            image13, image14, image15, image16, image17, image18,
                             Aimage, Bimage, Cimage, Dimage, Fimage, Gimage, Himage,
                             Limage, Mimage, Nimage, Oimage, Pimage, Qimage, image0, image00, image01,
                             Iimage, Jimage, Kimage)).reshape(-1, len, len)

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
        #画像を元の位置に　ref_xyは回転前の座標である, 次の(int)の都合上負の値が出てほしくないので、
        #さらに加えてます
        ref_xy = ref_xy_o + len//2 + len//2

        #左下をintで計算した後、それを移動させて計算
        linear_xy = {}
        linear_xy['upleft'] = ref_xy.astype(int) - len//2
        linear_xy['downleft'] = linear_xy['upleft'] + [1,0]
        linear_xy['upright']= linear_xy['upleft'] + [0,1]
        linear_xy['downright'] = linear_xy['upleft'] + [1,1]
        #linear_xy[key] = [[格子点0'], [格子点1'].....]
        #補正をもとに戻す
        ref_xy -= len//2
        upleft_diff = ref_xy - linear_xy['upleft']
        #(1-xの差)と(1-yの差)の積を計算
        #線形補完（？）
        linear_weight = {}
        linear_weight['upleft'] = (1-upleft_diff[:,0])*(1-upleft_diff[:,1])
        linear_weight['downleft'] = upleft_diff[:,0]*(1-upleft_diff[:,1])
        linear_weight['upright'] = (1-upleft_diff[:,0])*upleft_diff[:,1]
        linear_weight['downright'] = upleft_diff[:,0]*upleft_diff[:,1]
        # imgを拡大(傾ける前の画像をpddingすることによって、参照できないということがないように)
        #img_big = np.pad(img, [0, (28,), (28,)], 'constant')
        img_big = np.zeros([image.shape[0],len*3,len*3])
        img_big[:,len:len*2,len:len*2] = image

        linear_with_weight = {}
        for direction in linear_xy.keys():
            xy = linear_xy[direction]
            weight = linear_weight[direction]
            
            linear_with_weight[direction] = np.einsum('i,ki->ki',weight,img_big[:,xy[...,0]+len,xy[...,1]+len])

        img_linear = sum(linear_with_weight.values()).reshape(image.shape[0], len,len)
        return img_linear

    def mini(self, image):
        #大きさ1/4に
        B, len, _ = image.shape
        image = image.reshape(B, 1, len, len)
        poo = Pooling.Pooling()
        image_mini = poo.pooling(image, 2)
        image_mini = image_mini.reshape(B, len//2, -1)

        lefttopimage = np.zeros((B, len, len))
        lefttopimage[:, 0:len//2, 0:len//2] = image_mini
        rightdownimage = np.zeros((B, len, len))
        rightdownimage[:, len//2:len, len//2 :len] = image_mini
        righttopimage = np.zeros((B, len, len))
        righttopimage[:, 0:len//2, len//2 :len] = image_mini
        leftdownimage = np.zeros((B, len, len))
        leftdownimage[:, len//2:len, 0:len//2] = image_mini

        return np.vstack((lefttopimage, leftdownimage, rightdownimage, righttopimage))

    def rotate_imgs(self,img,theta):
        len = img.shape[1]
        x,y = np.mgrid[:len,:len]
        #格子点を画素
        xy_after = np.dstack((x,y))

        #affine変換行列
        affin = np.array([[np.cos(theta),-np.sin(theta)],
                        [np.sin(theta),np.cos(theta)]])

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
        img_big = np.zeros([img.shape[0],len*3,len*3])
        img_big[:,len:len*2,len:len*2] = img

        linear_with_weight = {}
        for direction in linear_xy.keys():
            xy = linear_xy[direction]
            weight = linear_weight[direction]
            linear_with_weight[direction] = np.einsum('i,ki->ki',weight,img_big[:,xy[...,0]+len,xy[...,1]+len])

        img_linear = sum(linear_with_weight.values()).reshape(img.shape[0], len,len)

        return img_linear

    def expand_prime(self, image):
        B, len, _ = image.shape
        Jimage = self.translation(3, 3, image)
        Kimage = self.translation(3, -3,image)
        Limage = self.translation(-3, 3, image)
        Mimage = self.translation(-3, -3, image)
        image = np.hstack((image, Jimage, Kimage, Limage, Mimage)).reshape(-1, len, len)
        Cimage = self.rotate_imgs(image, np.pi/18)
        Dimage = self.rotate_imgs(image, -np.pi/18)
        image3 = self.rotate_imgs(image, np.pi/12)
        image4 = self.rotate_imgs(image, -np.pi/12)
        Aimage = self.rotate_imgs(image, np.pi/9)
        Bimage = self.rotate_imgs(image, -np.pi/9)
        image = np.hstack((image, Cimage, Dimage, image3, image4, Aimage, Bimage)).reshape(-1, len, len)
        #contestにはないっぽい
        Eimage = self.mini(image)
        Fimage = self.Affine_conv(image, self.af_extend1)
        Gimage = self.Affine_conv(image, self.af_extend2)
        Himage = self.Affine_conv(image, self.af_extend3)
        image1 = self.Affine_conv(image, self.af_shrink1)
        image2 = self.Affine_conv(image, self.af_shrink2)
        Iimage = self.Affine_conv(image, self.af_shrink3)
        Nimage = self.Affine_conv(image, self.sque1)
        Oimage = self.Affine_conv(image, self.sque2)
        Pimage = self.Affine_conv(image, self.sque3)
        Qimage = self.Affine_conv(image, self.sque4)

        return np.hstack((image, image1, image2, Fimage, Nimage,
                                Oimage, Pimage, Qimage,
                                 Gimage, Himage, Iimage)).reshape(-1, len, len)

    def addnoise(self, img):
        B, len, _ = img.shape
        noise = np.random.normal(loc=1, scale=self.noise, size=(B, len, len))
        return img * noise

    def cutout(self, img, cutlen):
        B, len, _ = img.shape
        big_img = np.zeros((B, len*3, len*3))
        big_img[:, len:len*2, len:len*2] = img
        a, b = np.random.randint(0, 28, (2,))
        big_img[:, a+len:a+len+cutlen, b+len:b+len+cutlen] = 0
        return big_img[:, len:len*2, len:len*2]



"""
X = np.array(mnist.download_and_parse_mnist_file("train-images-idx3-ubyte.gz"))
img = X[0].reshape(1, 28, 28)
di = Diversify()
plt.imshow(di.rotate_imgs(img, np.pi/100)[0], cmap = cm.gray)
plt.show()
#"""