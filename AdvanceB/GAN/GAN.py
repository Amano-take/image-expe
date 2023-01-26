import numpy as np
# Keras 関係の import
import tensorflow.keras as keras
from tensorflow.keras import datasets, models, layers
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm


class GANmnist():

    def __init__(self, img_rows=28, img_cols=28, channel=1):

        self.img_rows, self.img_cols, self.channel = img_rows, img_cols, channel
        self.datagen = ImageDataGenerator(
            rotation_range = 15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range= 5,
            zoom_range=0.1,
            data_format='channels_last'
        )
        dopt = Adam(1e-4)
        opt = Adam(1e-5 * 3)

        self.discriminator = self.create_discriminator(dopt)
        self.generator = self.create_generator()
        self.trainable_or_not(self.discriminator)
        self.GAN = self.create_GAN(self.discriminator, self.generator, opt)
        self.train(50000)

    def trainable_or_not(self, net, trainable=False):
        net.trainable = trainable
        

    def create_generator(self):
        inputs = Input(shape=(100,))
        dim, depth, momentum, drate = 7, 256, 0.8, 0.5
    
        #1*1*100 -> 7*7*256
        x = Dense(dim*dim*depth)(inputs)
        x = BatchNormalization(momentum=momentum)(x)
        x = LeakyReLU(0.2)(x)
        x = Reshape([7, 7, 256])(x)
        x = Dropout(drate)(x)

        #7*7*256 -> 14*14*128
        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(128, (3, 3), padding='same')(x)
        x = BatchNormalization(momentum=momentum)(x)
        x = LeakyReLU()(x)

        #
        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(64, (3, 3), padding='same')(x)
        x = BatchNormalization(momentum=momentum)(x)
        x = LeakyReLU()(x)

        #
        x = Conv2D(32, (3, 3), padding='same')(x)
        x = BatchNormalization(momentum=momentum)(x)
        x = LeakyReLU()(x)

        #
        x = Conv2D(1, (1, 1), padding='same')(x)
        y = Activation('tanh')(x)

        generator = Model(inputs, y)
        return generator

    def create_discriminator(self, dopt):
        drate = 0.3

        input_shape = (self.img_rows, self.img_cols, self.channel)
        inputs = Input(shape=input_shape)

        x = Conv2D(256, (5, 5), strides=(2, 2),padding='same')(inputs)
        x = LeakyReLU(0.2)(x)
        x = Dropout(drate)(x)

        x = Conv2D(512, (5, 5), strides=(2, 2), padding='same')(x)
        x = LeakyReLU(0.2)(x)
        x = Dropout(drate)(x)

        x = Flatten()(x)
        x = Dense(256)(x)
        x = LeakyReLU(0.2)(x)
        x = Dropout(drate)(x)
        y = Dense(2, activation='softmax')(x)

        discriminator = Model(inputs, y)
        discriminator.compile(loss='binary_crossentropy', optimizer=dopt, metrics=['accuracy'])
        discriminator.summary()
        return discriminator

    def create_GAN(self, dis, gen, opt):
        input = Input(shape=(100,))
        x = gen(input)
        y = dis(x)
        GAN = Model(input, y)
        for l in GAN.layers:
            print(l.name, l.trainable)
        GAN.compile(loss='binary_crossentropy', optimizer=opt)
        GAN.summary()
        return GAN

    def train_discriminator(self, X_train, batch_size):
        image_batch = X_train[np.random.randint(0, X_train.shape[0], size=batch_size), :, :, :]
        noise_gen = np.random.normal(0, 1, size=[batch_size, 100])
        generated_images = self.generator.predict(noise_gen)

        Y1 = np.zeros([batch_size, 2])
        Y2 = np.zeros([batch_size, 2])
        #[1, 0] -> 偽物　[0, 1] -> 本物
        Y1[:, 1] = np.random.uniform(0.7, 1.0, size=(batch_size,))
        Y1[:, 0] = 1 - Y1[:, 1]
        Y2[:, 1] = np.random.uniform(0.0, 0.3, size=(batch_size,))
        Y2[:, 0] = 1 - Y1[:, 1]

        gen0 = self.datagen.flow(image_batch, Y1, batch_size=batch_size)
        gen1 = self.datagen.flow(generated_images, Y2, batch_size=batch_size)

        d_loss1 = self.discriminator.train_on_batch(next(gen0)[0], next(gen0)[1])
        d_loss2 = self.discriminator.train_on_batch(next(gen1)[0], next(gen1)[1])
        d_loss = [(d_loss1[0] + d_loss2[0]) / 2, (d_loss1[1] + d_loss2[1]) / 2 ]
        return d_loss

    def train_GAN(self, batch_size):
        noise_tr = np.random.normal(0, 1, size=[batch_size, 100])
        y = np.zeros([batch_size, 2])
        y[:, 1] = 1
        g_loss = self.GAN.train_on_batch(noise_tr, y)
        return g_loss

    def plot_loss(self, losses, filename="./AdvanceB/Image/loss1.png"):
        plt.figure(figsize=(10, 8))
        plt.plot(losses["d"], label='discriminitive loss')
        plt.plot(losses["g"], label='generative loss')
        plt.legend()
        plt.savefig(filename)
        plt.close('all')

    def plot_gen(self, noise, filename='result.png'):
        generated_images = self.generator.predict(noise)
        plt.figure(figsize=(10, 10))
        for i in range(generated_images.shape[0]):
            plt.subplot(4, 4, i+1)
            img = generated_images[i,:,:,:]
            img = np.reshape(img, [28, 28])
            plt.imshow(img, cmap='gray')
            plt.axis('off')
        plt.tight_layout()
        plt.savefig(filename)
        plt.close('all')

    def train(self, epochs, batch_size=130, save_interval = 3000):

        (X_train, _), (_, _) = keras.datasets.mnist.load_data()

        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)

        half_batch = int(batch_size / 2)
        noise_for_plot = np.random.normal(0, 1, size=[16, 100])


        losses = {"d" : [], "g" : []}

        for epoch in tqdm(range(epochs)):
            print()
            d_loss = self.train_discriminator(X_train, half_batch)
            g_loss = self.train_GAN(batch_size)

            if epoch % 100 == 0:
                losses['d'].append(d_loss[0])
                losses['g'].append(g_loss)

            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss) + "\033[2A")
        
            
            if epoch % save_interval == save_interval - 1:
                self.plot_loss(losses)
                self.plot_gen(noise_for_plot, "./AdvanceB/Image/resultB_%d.png" % (epoch+1))

            if d_loss[0] < 0.1:
                break
            
        self.GAN.save("./AdvanceB/GAN/GAN1.h5")

GANmnist()
