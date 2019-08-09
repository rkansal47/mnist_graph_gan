import setGPU
from keras.datasets import mnist
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, LeakyReLU, Dropout, Input
from keras.optimizers import Adam, RMSprop
from keras import initializers
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.cm as cm
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
# import tensorflow as tf


from os import listdir
from os.path import isfile, join
from tqdm import tqdm
# from mbdiscriminator import MinibatchDiscrimination

K.set_image_dim_ordering('th')

np.random.seed(1000)

(X_train_pre, Y_train), (X_test, Y_test) = mnist.load_data()

#converting to single vector and normalized
X_train_pre = (X_train_pre.astype(np.float32)-127.5)/255

X_train_pre.shape

X_train = []

imrange = np.linspace(-0.5, 0.5, num=28)

xs, ys = np.meshgrid(imrange, imrange)

xs = xs.reshape(-1)
ys = ys.reshape(-1)

X_train=np.array(list(map(lambda x: np.array([xs, ys, x.reshape(-1)]).T, X_train_pre)))

print(X_train.shape)

gen_in_dim = 100 #size of random input vector for GAN
output_dim = X_train.shape[1]*X_train.shape[2]
batch_size = 128
batches_per_epoch = int(X_train.shape[0]/batch_size)
learning_rate = 0.0002
disc_dropout = 0.3

output_dim

examples_noise = np.random.normal(0, 1, size=[100, gen_in_dim])

latest_model_epoch = 180

def init_models():
    adam = Adam(lr=learning_rate, beta_1 = 0.5)

    generator = load_model("models/8_new_dispgan_generator_epoch_%d.h5" % latest_model_epoch)
    discriminator = load_model("models/8_new_dispgan_discriminator_epoch_%d.h5" % latest_model_epoch)

    discriminator.trainable = False
    ganInput = Input(shape=(gen_in_dim,))
    x = generator(ganInput)
    ganOutput = discriminator(x)
    gan = Model(inputs=ganInput, outputs=ganOutput)
    gan.compile(loss='binary_crossentropy', optimizer=adam)

    return generator, discriminator, gan


def init_models_2():

    adam = Adam(lr=learning_rate, beta_1 = 0.5)

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

    # generator.add(Dense(2048))
    # generator.add(LeakyReLU(alpha=0.2))

    generator.add(Dense(2560))
    generator.add(LeakyReLU(alpha=0.2))

    generator.add(Dense(output_dim, activation='tanh'))
    generator.compile(optimizer=adam, loss='binary_crossentropy')

    discriminator = Sequential()

    discriminator.add(Dense(2560, input_dim=output_dim, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    discriminator.add(LeakyReLU(alpha=0.2))
    discriminator.add(Dropout(disc_dropout))

    discriminator.add(Dense(1024))
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

def disp_sample_outputs(generator, num_cols, num_rows, name, epoch, dlosses, glosses):

    fig = plt.figure(figsize=(10,10))
    rand_in = np.random.normal(0, 1, size=[num_cols*num_rows, gen_in_dim])
    gen_out = generator.predict(rand_in).reshape(num_cols*num_rows, 784, 3)*[28, 28, 1]+[14, 14, 0]

    for i in range(1, 100+1):
        fig.add_subplot(10, 10, i)
        im_disp = np.zeros((28,28)) - 0.5
        for x in gen_out[i-1]:
            if(x[1] > 27):
                x[1] = 27
            if(x[1] < 0):
                x[1] = 0

            if(x[0] > 27):
                x[0] = 27
            if(x[0] < 0):
                x[0] = 0

            im_disp[min(27, int(np.round(x[1]))), min(27, int(np.round(x[0])))] = x[2]
        plt.imshow(im_disp, cmap=cm.gray_r, interpolation='nearest')
        plt.axis('off')

    plt.savefig("figs/"+name + "_" + str(epoch) + ".png")

    plt.figure()
    plt.plot(dlosses)
    plt.savefig("losses/"+name + "_disc_" + str(epoch) + ".png")

    plt.figure()
    plt.plot(glosses)
    plt.savefig("losses/"+name + "_gan_" + str(epoch) + ".png")

    plt.show()

def train_gan(generator, discriminator, gan, epochs, num_saves, name):
    gan_loss = []
    disc_loss = []
    # epochs_between_rows = int(float(epochs)/num_saves)
    for i in range(latest_model_epoch, epochs):
        disp_sample_outputs(generator, 10, 10, name, i, disc_loss, gan_loss)

        if(i%20==0):
            save_models(generator, discriminator, name, i)

        print("Epoch ", i)
        for _ in tqdm(range(batches_per_epoch)):
            noise = np.random.normal(0, 1, size=[batch_size, gen_in_dim])
            imageBatch = X_train[np.random.randint(0, X_train.shape[0], size=batch_size)].reshape(batch_size, -1)

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
            discriminator.trainable = False
            gbatch_loss = gan.train_on_batch(noise, np.ones(batch_size))

        disc_loss.append(dbatch_loss)
        gan_loss.append(gbatch_loss)

    disp_sample_outputs(generator, 10, 10, name, epochs)
    save_models(generator, discriminator, name, epochs)

    return (disc_loss, gan_loss)

def run_trial(name, epochs):
    onlyfiles = [f for f in listdir('figs/') if isfile(join('figs/', f))]
    if (name + "_0.png" in onlyfiles):
        print("file name already used")
        # return
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

run_trial("8_new_disp", 300)
