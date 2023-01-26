import numpy as np
# Keras 関係の import
import tensorflow.keras as keras
from tensorflow.keras import datasets, models, layers
# GPGPU リソースを全消費してしてしまう問題の回避
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from Layer.ConAn import ConAn


physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
        print('{} memory growth: {}'.format(device, tf.config.experimental.get_memory_growth(device)))
else:
    print("Not enough GPU hardware devices available")

def addnoise(img):
    noise = np.random.normal(loc=1, scale=0.01, size=img.shape)
    return img * noise

#データジェネレーター
def data_creater(bo = True, scale = 0.01, rotation_range = 15, width_shift_range = 0.15, shear_range=5, zoom_range=0.1):
    if bo :
        datagen  = ImageDataGenerator(
            rotation_range = rotation_range,
            width_shift_range=0.15,
            height_shift_range=0.15,
            shear_range= shear_range,
            zoom_range=0.1,
            preprocessing_function=addnoise,
            data_format='channels_last'
        )
    else :
        datagen  = ImageDataGenerator(
            rotation_range = 15,
            width_shift_range=0.15,
            height_shift_range=0.15,
            shear_range= 5,
            zoom_range=0.1,
            data_format='channels_last'
        )
    
    return datagen

# MNIST データの準備
img_rows, img_cols = 28, 28 # 画像サイズは 28x28
num_classes = 10 # クラス数
l2lambda = 0.002
(X, Y), (Xtest, Ytest) = keras.datasets.mnist.load_data() # 訓練用とテスト（兼 validation）用のデータを取得
X = np.vstack((X, Xtest))
Y = np.hstack((Y, Ytest))
contestX, contestY = ConAn.get(0, 200)

contestX = contestX.reshape(contestX.shape[0], img_rows, img_cols, 1)
X = X.reshape(X.shape[0],img_rows,img_cols,1)
 # X を (画 像 ID，28, 28, 1) の 4 次元配列に変換
Xtest = Xtest.reshape(Xtest.shape[0],img_rows,img_cols,1)
contestX = contestX.astype('float32') / 255.0
X = X.astype('float32') / 255.0 # 各画素の値を 0～1 に正規化
Xtest = Xtest.astype('float32') /255.0
input_shape = (img_rows, img_cols, 1)
Y = keras.utils.to_categorical(Y, num_classes) # one-hot-vector へ変換
Ytest1 = keras.utils.to_categorical(Ytest, num_classes)
contestY  = keras.utils.to_categorical(contestY, num_classes)

# モデルの定義
def create_model(i = 1, ch=64, l2lambda=0.0001, kernel_size=(5,5), drate=0.5, pool_size=(2, 2)):
    model = models.Sequential()
    # 3x3 の畳み込み層．出力は 32 チャンネル．活性化関数に ReLU．入力データのサイ
    #ズは input_shape で指定．入出力のサイズ（row と col）が同じになるように設定．
    model.add(layers.Conv2D(ch, kernel_size= kernel_size, activation='relu',
    input_shape=input_shape, padding='same'))
    model.add(layers.MaxPooling2D(pool_size=pool_size))
    model.add(layers.BatchNormalization())

    # 入力を 1 次元配列に並び替える
    model.add(layers.Flatten())
    model.add(layers.Dropout(drate))
    #batch-normalize
    # 全結合層．出力ノード数は num_classes（クラス数）．活性化関数に softmax．
    model.add(layers.Dense(num_classes, activation='softmax', kernel_regularizer=keras.regularizers.l2(l2lambda), bias_regularizer=keras.regularizers.l2(l2lambda)))

    return model



opt3 = keras.optimizers.RMSprop(epsilon=1e-8)
epochs = 50
batch_size = 100
history = []

for i in range(6):
    if i % 2 == 0:
        sr = 5
    else:
        sr = 10
    if i % 3 == 0:
        rr = 10
    elif i % 3 == 1:
        rr = 20
    else:
        rr = 30

    model = create_model()
    datagen = data_creater(rotation_range=rr, shear_range=sr)
    model.compile(
    loss=keras.losses.categorical_crossentropy,
    optimizer=opt3, metrics=['acc'])
    result = model.fit_generator(datagen.flow(X, Y, batch_size=128),
        steps_per_epoch=100, epochs = epochs, validation_data=(contestX, contestY))
    """result = model.fit(X,Y, batch_size=batch_size,
    epochs=epochs, validation_data=(Xtest,Ytest1), steps_per_epoch=100)"""
    history.append(result.history)


fig = plt.figure()
for i in range(6):
    plt.plot(history[i]['loss'], label=str(i))
    plt.legend()
    plt.savefig("./contest/forme/loss_history_l2.png")

fig = plt.figure()
for i in range(6):
    plt.plot(history[i]['val_loss'], label=str(i))
    plt.legend()
    plt.savefig("./contest/forme/val_loss_history_l2.png")


"""
plt.plot(history['loss'], label='loss') # 教師データの損失
plt.plot(history['val_loss'], label='val_loss') # テストデータの損失
plt.legend()
plt.savefig("./AdvanceB/Image/loss_history_ch.png")
fig = plt.figure()
plt.plot(history['acc'], label='acc') # 教師データでの精度
plt.plot(history['val_acc'], label='val_acc') # テストデータでの精度
plt.legend()
plt.savefig("./AdvanceB/Image/loss_acc_ch.png") 
"""