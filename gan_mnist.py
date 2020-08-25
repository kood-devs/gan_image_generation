"""
    GANによるMNIST画像生成
    参考：https://book.mynavi.jp/ec/products/detail/id=113324
"""

import matplotlib.pyplot as plt
import numpy as np

from keras.datasets import mnist
from keras.layers import Dense, Flatten, Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential
from keras.optimizers import Adam


# パラメタ
# 画像データのパラメタ
img_rows = 28
img_cols = 28
channels = 1
img_shape = (img_rows, img_cols, channels)
z_dim = 100

# 学習に関するパラメタ
iterations = 20000
batch_size = 128
sample_interval = 1000


def build_generator(img_shape, z_dim):
    # 生成器を作成する関数
    model = Sequential()
    model.add(Dense(128, input_dim=z_dim))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dense(img_rows*img_cols*channels, activation='tanh'))
    model.add(Reshape(img_shape))
    return model


def build_discrimiantor(img_shape):
    # 識別器を作成する関数
    model = Sequential()
    model.add(Flatten(input_shape=img_shape))
    model.add(Dense(128))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dense(1, activation='sigmoid'))
    return model


def build_gan(generator, discriminator):
    # 生成器と識別器を結合する関数
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    return model


def sample_images(generator, image_grid_rows=4, image_grid_columns=4):
    # 途中経過を表示する関数
    z = np.random.normal(0, 1, (image_grid_rows * image_grid_columns, z_dim))
    gen_imgs = generator.predict(z)
    gen_imgs = 0.5 * gen_imgs + 0.5

    fig, axs = plt.subplots(image_grid_rows,
                            image_grid_columns,
                            figsize=(4, 4),
                            sharey=True,
                            sharex=True)

    cnt = 0
    for i in range(image_grid_rows):
        for j in range(image_grid_columns):
            axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
            axs[i, j].axis('off')
            cnt += 1


def main():
    # 識別器を準備
    discriminator = build_discrimiantor(img_shape)
    discriminator.compile(loss='binary_crossentropy',
                          optimizer=Adam(),
                          metrics=['accuracy'])

    # 生成器を準備
    generator = build_generator(img_shape, z_dim)
    discriminator.trainable = False

    # GANモデルを準備
    gan = build_gan(generator, discriminator)
    gan.compile(loss='binary_crossentropy', optimizer=Adam())

    # 結果を格納するリストを用意
    losses = []
    accuracies = []
    iteration_checkpoints = []

    # データを準備
    (x_train, _), (_, _) = mnist.load_data()

    x_train = x_train / 127.5 - 1.0
    x_train = np.expand_dims(x_train, axis=3)

    real = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    # 学習を実施
    for iteration in range(iterations):
        idx = np.random.randint(0, x_train.shape[0], batch_size)
        imgs = x_train[idx]

        z = np.random.normal(0, 1, (batch_size, 100))
        gen_imgs = generator.predict(z)

        d_loss_real = discriminator.train_on_batch(imgs, real)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
        d_loss, accuracy = 0.5 * np.add(d_loss_real, d_loss_fake)

        z = np.random.normal(0, 1, (batch_size, 100))
        gen_imgs = generator.predict(z)

        g_loss = gan.train_on_batch(z, real)

        if (iteration + 1) % sample_interval == 0:
            # 途中経過を表示
            losses.append((d_loss, g_loss))
            accuracies.append(100.0 * accuracy)
            iteration_checkpoints.append(iteration + 1)

            print('%d [D loss: %f, acc.: %.2f%%] [G loss: %f]' %
                  (iteration + 1, d_loss, 100.0 * accuracy, g_loss))
            sample_images(generator)

    # 比較用にオリジナルの画像を表示
    fig = plt.figure(figsize=(8, 8))
    fig.subplots_adjust(left=0, right=1, bottom=0,
                        top=1.0, hspace=0.05, wspace=0.05)
    for i in range(8 * 8):
        ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
        ax.imshow(x_train[i].reshape((28, 28)), cmap='gray')
    plt.show()


if __name__ == '__main__':
    main()
