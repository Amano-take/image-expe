import numpy as np
# Keras 関係の import
import tensorflow.keras as keras
from tensorflow.keras import datasets, models, layers
# GPGPU リソースを全消費してしてしまう問題の回避
import tensorflow as tf
import matplotlib.pyplot as plt

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
        print('{} memory growth: {}'.format(device, tf.config.experimental.get_memory_growth(device)))
else:
    print("Not enough GPU hardware devices available")

img_rows, img_cols = 28, 28 # 画像サイズは 28x28
num_classes = 10
(X, Y), (Xtest, Ytest) = keras.datasets.mnist.load_data() # 訓練用とテスト（兼 validation）用のデータを取得
X = X.reshape(X.shape[0],img_rows,img_cols,1)
 # X を (画 像 ID，28, 28, 1) の 4 次元配列に変換
Xtest = Xtest.reshape(Xtest.shape[0],img_rows,img_cols,1)
X = X.astype('float32') / 255.0 # 各画素の値を 0～1 に正規化
Xtest = Xtest.astype('float32') /255.0
input_shape = (img_rows, img_cols, 1)
Y = keras.utils.to_categorical(Y, num_classes) # one-hot-vector へ変換
Ytest1 = keras.utils.to_categorical(Ytest, num_classes)


model = models.Sequential()
# 3x3 の畳み込み層．出力は 32 チャンネル．活性化関数に ReLU．入力データのサイ
#ズは input_shape で指定．入出力のサイズ（row と col）が同じになるように設定．
model.add(layers.Conv2D(32, kernel_size=(3,3), activation='relu',
input_shape=input_shape, padding='same'))
# 3x3 の畳み込み層．出力は 64 チャンネル．活性化関数に ReLU．入力データのサイ
#ズは自動的に決まるので設定不要．
model.add(layers.Conv2D(64,(3,3),activation='relu',padding='same'))
# 2x2 の最大値プーリング
model.add(layers.MaxPooling2D(pool_size=(2,2)))
# 入力を 1 次元配列に並び替える
model.add(layers.Flatten())
# 全結合層．出力ノード数は 128．活性化関数に ReLU．
model.add(layers.Dense(128,activation='relu'))
# 全結合層．出力ノード数は num_classes（クラス数）．活性化関数に softmax．
model.add(layers.Dense(num_classes, activation='softmax'))
# 作成したモデルの概要を表示
print (model.summary()) 

model.compile(
loss=keras.losses.categorical_crossentropy,
optimizer=keras.optimizers.SGD(momentum=0.9), metrics=['acc'])

epochs = 3
batch_size = 32

result = model.fit(X,Y, batch_size=batch_size,
epochs=epochs, validation_data=(Xtest,Ytest1))

history = result.history

fig = plt.figure()
plt.plot(history['loss'], label='loss') # 教師データの損失
plt.plot(history['val_loss'], label='val_loss') # テストデータの損失
plt.legend()
plt.savefig("./AdvanceB/Image/loss_history_withaug.png")
fig = plt.figure()
plt.plot(history['acc'], label='acc') # 教師データでの精度
plt.plot(history['val_acc'], label='val_acc') # テストデータでの精度
plt.legend()
plt.savefig("./AdvanceB/Image/loss_acc_withaug.png") 