import numpy as np
import matplotlib.pyplot as plt
from pylab import cm
import Imshow
import mnist

class Initialize():
    def __init__(self, X, Y, B, C, p):
        #Bはtestを割り切る数を想定
        #->convがうまくいかない
        self.X = X
        self.Y = Y
        self.B = B
        self.C = C
        self.p = p

    def onehot(self, answer):
        onehot = np.zeros((answer.size, 10))
        onehot[np.arange(answer.size), answer] = 1
        onehot = onehot.reshape(self.B, self.C, 1)
        return onehot

    def randomselect(self):
        batch_random = np.random.randint(0, self.X.shape[0], self.B)
        # 画像取得
        Batch_img = np.array(self.X[batch_random])
        # 正解を取得
        answer = np.array(self.Y[batch_random])
        #plt.imshow(Batch_img[0], cmap=cm.gray)
        #plt.show()
        #print(answer[0])
        # 正解をone-hot vectorに
        onehot = self.onehot(answer)
        return Batch_img, onehot, answer

    def randomselect_withonehot(self):
        batch_random = np.random.randint(0, self.X.shape[0], self.B)
        # 画像取得
        Batch_img = np.array(self.X[batch_random])
        # 正解を取得
        onehot = np.array(self.Y[batch_random])

        return Batch_img, onehot

    def randomselect_with_arg(self):
        batch_random = np.random.randint(0, self.X.shape[0], self.B)
        # 画像取得
        Batch_img = np.array(self.X[batch_random])
        # 正解を取得
        answer = np.array(self.Y[batch_random])
        #plt.imshow(Batch_img[0], cmap=cm.gray)
        #plt.show()
        #print(answer[0])
        # 正解をone-hot vectorに
        return Batch_img, answer, batch_random

    def orderselect(self, i):
        Batch_img = self.X[i*self.B : (i+1)*self.B]
        answer = self.Y[i*self.B : (i+1)*self.B]
        return Batch_img, answer


    def orderselect_test(self, i):
        Batch_img = self.X[i*self.B : (i+1)*self.B]
        return Batch_img

    def labelSmoothing(self):
        batch_random = np.random.randint(0, self.X.shape[0], self.B)
        # 画像取得
        Batch_img = np.array(self.X[batch_random])
        # 正解を取得
        answer = np.array(self.Y[batch_random])
        #plt.imshow(Batch_img[0], cmap=cm.gray)
        #plt.show()
        #print(answer[0])
        # 正解をsmoothing vector に
        smooth = self.smooth(answer, self.p)
        return Batch_img, smooth, answer

    def smooth(self, answer, eps):
        onehot = np.zeros((answer.size, 10))
        smooth = onehot + eps/9
        smooth[np.arange(answer.size), answer] = 1 - eps
        smooth = smooth.reshape(self.B, self.C, 1)
        return smooth

    def mixupselect(self, num, beta):
        batch_random = np.random.randint(0, self.X.shape[0], self.B)
        # 画像取得
        Batch_img = np.array(self.X[batch_random])
        # 正解を取得
        answer = np.array(self.Y[batch_random])
        #plt.imshow(Batch_img[0], cmap=cm.gray)
        #plt.show()
        #print(answer[0])
        # 正解をone-hot vectorに
        onehot = self.onehot(answer)
        mixdate, mixdate2, mixans, mixans2 = self.mixup(Batch_img[0:num*2], onehot[0:num*2], beta)
        Batch_img = np.vstack((mixdate, mixdate2, Batch_img[num*2:self.B]))
        onehot = np.vstack((mixans, mixans2, onehot[num*2:self.B]))
        return Batch_img, onehot, answer
    
    def mixup(self, img, one, beta):
        num = img.shape[0] // 2
        plambda = np.random.beta(beta, beta, (1,))
        imgA = img[0:num]
        imgB = img[num:]
        oneA = one[0:num]
        oneB = one[num:]
        mixdate = (1 - plambda) * imgA + plambda * imgB
        ans = (1-plambda) * oneA + plambda * oneB
        mixdate2 = (1-plambda) * imgB + plambda * imgA
        ans2 = (1-plambda) * oneB + plambda * oneA
        return mixdate, mixdate2, ans, ans2


"""
X = np.array(mnist.download_and_parse_mnist_file("train-images-idx3-ubyte.gz"))
Y = np.array(mnist.download_and_parse_mnist_file("train-labels-idx1-ubyte.gz"))
ini = Initialize(X, Y, 100, 10, 0.1)
A, B, _ = ini.mixupselect(20, 10)
print(A.shape, B.shape)
Imshow.Imshow.imansshow(A, B)

#"""

