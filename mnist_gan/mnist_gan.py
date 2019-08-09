from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, LeakyReLU, Dropout, Input
from keras.optimizers import Adam
from keras import initializers
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
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

examples_noise = np.random.normal(0, 1, size=[100, gen_in_dim])

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
    gan.compile(loss='binary_crossentropy', optimizer=adam)

    return generator, discriminator, gan

def save_models(generator, discriminator, name, epoch):
    generator.save('models/%sgan_generator_epoch_%d.h5' % (name, epoch))
    discriminator.save('models/%sgan_discriminator_epoch_%d.h5' % (name, epoch))

def disp_sample_outputs(generator, num_cols, num_rows, name, epoch):
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

def train_gan(generator, discriminator, gan, epochs, num_saves, name):
    gan_loss = []
    disc_loss = []
    epochs_between_rows = int(float(epochs)/num_saves)
    for i in range(epochs):
        disp_sample_outputs(generator, 10, 10, name, i)

        if(i%20==0):
            save_models(generator, discriminator, name, i)

        print("Epoch ", i)
        for _ in tqdm(range(batches_per_epoch)):
            noise = np.random.normal(0, 1, size=[batch_size, gen_in_dim])
            imageBatch = X_train[np.random.randint(0, X_train.shape[0], size=batch_size)]

            # Generate fake MNIST images
            generatedImages = generator.predict(noise)
            # print np.shape(imageBatch), np.shape(generatedImages)
            X = np.concatenate([imageBatch, generatedImages])

            # Labels for generated and real data
            yDis = np.zeros(2*batch_size)
            # One-sided label smoothing
            yDis[:batch_size] = 0.9

            # Train discriminator
            discriminator.trainable = True
            dbatch_loss = discriminator.train_on_batch(X, yDis)

            # Train generator
            noise = np.random.normal(0, 1, size=[batch_size, gen_in_dim])
            yGen = np.ones(batch_size)
            discriminator.trainable = False
            gbatch_loss = gan.train_on_batch(noise, yGen)

        disc_loss.append(dbatch_loss)
        gan_loss.append(gbatch_loss)

    disp_sample_outputs(generator, 10, 10, name, epochs)
    save_models(generator, discriminator, name, epochs)

    return (disc_loss, gan_loss)

def run_trial(name, epochs):
    onlyfiles = [f for f in listdir('figs/') if isfile(join('figs/', f))]
    if (name + ".png" in onlyfiles):
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

run_trial("49_200_epochs", 200)
