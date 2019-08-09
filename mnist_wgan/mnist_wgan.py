from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, LeakyReLU, Dropout, Input
from keras.optimizers import Adam
from keras import initializers
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.cm as cm
import os
os.environ["KERAS_BACKEND"] = "tensorflow"

from os import listdir
from os.path import isfile, join
from tqdm import tqdm
# from mbdiscriminator import MinibatchDiscrimination

K.set_image_dim_ordering('th')

np.random.seed(1000)

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

#converting to single vector and normalized
X_train = (X_train.astype(np.float32).reshape(X_train.shape[0], -1)-127.5)/255

def plot_num(i):
    plt.imshow(X_train[i].reshape(28, 28), cmap=cm.gray)
    plt.show()

gen_in_dim = 100 #size of random input vector for GAN
img_dim = X_train.shape[1]
batch_size = 128
batches_per_epoch = int(X_train.shape[0]/batch_size)
learning_rate_gan = 0.0002
learning_rate_disc = 0.0002
gen_dropout = 0.3
disc_dropout = 0.3
clip_value = 0.01
num_critic = 1

examples_noise = np.random.normal(0, 1, size=[100, gen_in_dim])

def wasserstein_loss(y_true, y_pred):
    return -K.mean(y_true*y_pred)

def init_models():

    adam = Adam(lr=learning_rate_gan, beta_1 = 0.5)

    generator = Sequential()

    generator.add(Dense(256, input_dim=gen_in_dim, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    generator.add(LeakyReLU(alpha=0.2))
    # generator.add(Dropout(gen_dropout))

    generator.add(Dense(512))
    generator.add(LeakyReLU(alpha=0.2))
    # generator.add(Dropout(gen_dropout))

    generator.add(Dense(1024))
    generator.add(LeakyReLU(alpha=0.2))
    # generator.add(Dropout(gen_dropout))

    generator.add(Dense(img_dim, activation='tanh'))
    generator.compile(optimizer=adam, loss='binary_crossentropy')

    discriminator = Sequential()

    discriminator.add(Dense(1024, input_dim=img_dim, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    discriminator.add(LeakyReLU(alpha=0.2))
    discriminator.add(Dropout(disc_dropout))

    discriminator.add(Dense(512))
    discriminator.add(LeakyReLU(alpha=0.2))
    discriminator.add(Dropout(disc_dropout))

    # discriminator.add(MinibatchDiscrimination(5, 3))

    discriminator.add(Dense(256))
    discriminator.add(LeakyReLU(alpha=0.2))
    discriminator.add(Dropout(disc_dropout))

    discriminator.add(Dense(1, activation='sigmoid')) #binary classification (real or fake = 1 or 0 respectively)
    discriminator.compile(optimizer=adam, loss='binary_crossentropy')

    # creating gan
    discriminator.trainable = False
    ganInput = Input(shape=(gen_in_dim,))
    x = generator(ganInput)
    ganOutput = discriminator(x)
    gan = Model(inputs=ganInput, outputs=ganOutput)
    gan.compile(loss=wasserstein_loss, optimizer=adam)

    return generator, discriminator, gan

def save_models(generator, discriminator, name, epoch):
    generator.save('models/%sgan_generator_epoch_%d.h5' % (name, epoch))
    discriminator.save('models/%sgan_discriminator_epoch_%d.h5' % (name, epoch))

def disp_sample_outputs(generator, num_cols, num_rows, name, epoch, dlosses, glosses):
    fig = plt.figure(figsize=(10,10))
    rand_in = np.random.normal(0, 1, size=[num_cols*num_rows, gen_in_dim])
    gen_out = generator.predict(rand_in).reshape(num_cols*num_rows, 28, 28)
    print(gen_out[0].reshape(784))
    for i in range(1, num_cols*num_rows+1):
        fig.add_subplot(num_rows, num_cols, i)
        plt.imshow(gen_out[i-1], cmap=cm.gray_r, interpolation='nearest')
        plt.axis('off')
    plt.savefig("figs/"+name + "_" + str(epoch) + ".png")
    plt.show()

    plt.figure()
    plt.plot(dlosses)
    plt.savefig("losses/"+name + "_disc_" + str(epoch) + ".png")

    plt.figure()
    plt.plot(glosses)
    plt.savefig("losses/"+name + "_gan_" + str(epoch) + ".png")

def train_gan(generator, discriminator, gan, epochs, num_saves, name):
    gan_loss = []
    disc_loss = []
    # epochs_between_rows = int(float(epochs)/num_saves)

    y_real = np.ones(batch_size)
    y_gen = -np.ones(batch_size)
    for i in range(epochs):
        disp_sample_outputs(generator, 10, 10, name, i, disc_loss, gan_loss)

        if(i%20==0):
            save_models(generator, discriminator, name, i)

        noise = np.random.normal(0, 1, size=[batch_size, gen_in_dim])
        gen_batch= generator.predict(noise)
        image_batch = X_train[np.random.randint(0, X_train.shape[0], size=batch_size)]

        print("disc prediction")
        print(discriminator.predict(np.concatenate([image_batch, gen_batch])))

        print("Epoch ", i)
        for _ in tqdm(range(batches_per_epoch)):

            dbatch_loss_real = 0
            dbatch_loss_fake = 0
            # Train discriminator
            discriminator.trainable = True

            for i in range(num_critic):
                noise = np.random.normal(0, 1, size=[batch_size, gen_in_dim])
                gen_batch= generator.predict(noise)
                image_batch = X_train[np.random.randint(0, X_train.shape[0], size=batch_size)]

                dbatch_loss_real += discriminator.train_on_batch(image_batch, y_real)
                dbatch_loss_fake += discriminator.train_on_batch(gen_batch, y_gen)

            for l in discriminator.layers:
                weights = l.get_weights()
                weights = [np.clip(w, -clip_value, clip_value) for w in weights]
                l.set_weights(weights)

            # Train generator
            noise = np.random.normal(0, 1, size=[batch_size, gen_in_dim])
            discriminator.trainable = False
            gbatch_loss = gan.train_on_batch(noise, y_real)

        disc_loss.append((dbatch_loss_real+dbatch_loss_fake)/num_critic)
        gan_loss.append(gbatch_loss)

    disp_sample_outputs(generator, 10, 10, name, epochs)
    save_models(generator, discriminator, name, epochs)

    return (disc_loss, gan_loss)

def run_trial(name, epochs):
    onlyfiles = [f for f in listdir('figs/') if isfile(join('figs/', f))]
    if (name + "_0.png" in onlyfiles):
        print("file name already used")
        return
    name = name
    generator, discriminator, gan = init_models()
    generator.summary()
    discriminator.summary()
    gan.summary()
    disc_loss, gan_loss = train_gan(generator, discriminator, gan, epochs, epochs, name)
    plt.figure()
    plt.plot(disc_loss)
    # plt.ylim((0,10))
    plt.savefig("figs/" + name + "_disc_loss.png")
    plt.show()

    plt.figure()
    plt.plot(gan_loss)
    # plt.ylim((0,10))
    plt.savefig("figs/" + name + "_gan_loss.png")
    plt.show()

    print(disc_loss)
    print(gan_loss)

run_trial("14_test_disc_predictions_sig", 200)
