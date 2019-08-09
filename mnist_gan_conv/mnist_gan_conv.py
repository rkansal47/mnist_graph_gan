# import os
# os.environ["KERAS_BACKEND"] = "tensorflow"
# # import tensorflow as tf
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
#
# # The GPU id to use, usually either "0" or "1";
# os.environ["CUDA_VISIBLE_DEVICES"]="1";


import setGPU
from keras.datasets import mnist
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, LeakyReLU, Dropout, Input, Conv2DTranspose, MaxPool1D, Flatten, Reshape, UpSampling1D, Conv1D
from keras.optimizers import Adam, RMSprop
from keras import initializers
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.cm as cm
# import os

from os import listdir
from os.path import isfile, join
from tqdm import tqdm
# from mbcritic import MinibatchDiscrimination
#
# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# K.set_session(sess)
#
# print(K.tensorflow_backend._get_available_gpus())
#
K.set_image_dim_ordering('th')

np.random.seed(1000)
num_thresholded = 784 #must be divisible by 4

(X_train_pre, Y_train), (X_test, Y_test) = mnist.load_data()

#converting to single vector and normalized
X_train_pre = (X_train_pre.astype(np.float32)-127.5)/255

# X_train_pre = X_train_pre[np.where(Y_train == 3)]

X_train = []

imrange = np.linspace(-0.5, 0.5, num=28, endpoint=False)

xs, ys = np.meshgrid(imrange, imrange)

xs = xs.reshape(-1)
ys = ys.reshape(-1)

X_train=np.array(list(map(lambda x: np.array([xs, ys, x.reshape(-1)]).T, X_train_pre)))
X_train=np.array(list(map(lambda x: x[x[:,2].argsort()][-num_thresholded:], X_train)))

print(X_train.shape)


# def disp_sample(i):
#     sample = X_train[i].reshape(num_thresholded, 3)*[28, 28, 1]+[14, 14, 0]
#     im_disp = np.zeros((28,28)) - 0.5
#     for x in sample:
#         im_disp[min(27, int(np.round(x[1]))), min(27, int(np.round(x[0])))] = x[2]
#
#     plt.imshow(im_disp, cmap=cm.gray_r, interpolation='nearest')
#     # plt.axis('off')
#     plt.show()
#
#
# disp_sample(140)

gen_in_dim = 100 #size of random input vector for GAN
output_dim = X_train.shape[1]*X_train.shape[2]
batch_size = 128
batches_per_epoch = int(X_train.shape[0]/batch_size)
learning_rate = 0.0002
disc_dropout = 0.3
clip_value = 0.01
num_critic = 5

lr_disc = 0.000

output_dim

examples_noise = np.random.normal(0, 1, size=[100, gen_in_dim])

def wasserstein_loss(y_true, y_pred):
    return -K.mean(y_true*y_pred)

def init_models_pre_trained():
    adam = Adam(lr=learning_rate, beta_1 = 0.5)

    generator = load_model("models/1_test_gan_generator_epoch_20.h5", custom_objects={'wasserstein_loss': wasserstein_loss})
    critic = load_model("models/1_test_gan_critic_epoch_20.h5", custom_objects={'wasserstein_loss': wasserstein_loss})

    critic.trainable = False
    ganInput = Input(shape=(gen_in_dim,))
    x = generator(ganInput)
    ganOutput = critic(x)
    gan = Model(inputs=ganInput, outputs=ganOutput)
    gan.compile(loss=wasserstein_loss, optimizer=adam)

    return generator, critic, gan

def init_models():

    adam = Adam(lr=learning_rate, beta_1 = 0.5)
    rmsprop = RMSprop(lr=learning_rate)

    optim = adam

    generator = Sequential()

    generator.add(Dense(int(num_thresholded/4), input_dim=gen_in_dim, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    generator.add(LeakyReLU(alpha=0.2))

    generator.add(Reshape((-1, 1)))

    generator.add(Conv1D(64, 3, padding='same'))
    generator.add(LeakyReLU(alpha=0.2))

    generator.add(UpSampling1D(size=2))
    #Size should be 392 now

    generator.add(Conv1D(32, 3, padding='same'))
    generator.add(LeakyReLU(alpha=0.2))

    generator.add(UpSampling1D(size=2))

    generator.add(Conv1D(3, 3, padding='same', activation='tanh'))

    generator.compile(optimizer=optim, loss=wasserstein_loss)

    critic = Sequential()

    critic.add(Conv1D(32, kernel_size=3, input_shape=(num_thresholded,3), padding='same'))
    critic.add(LeakyReLU(alpha=0.2))
    # critic.add(Dropout(0.3))

    critic.add(Conv1D(64, kernel_size=3, padding='same'))
    critic.add(LeakyReLU(alpha=0.2))

    critic.add(MaxPool1D(pool_size=2))
    critic.add(Dropout(0.3))

    critic.add(Flatten())

    critic.add(Dense(128))
    critic.add(LeakyReLU(alpha=0.2))
    critic.add(Dropout(0.5))

    critic.add(Dense(1))

    critic.compile(optimizer=optim, loss=wasserstein_loss)

    # creating gan
    critic.trainable = False
    ganInput = Input(shape=(gen_in_dim,))
    x = generator(ganInput)
    ganOutput = critic(x)
    gan = Model(inputs=ganInput, outputs=ganOutput)
    gan.compile(loss=wasserstein_loss, optimizer=optim)

    return generator, critic, gan

def save_models(generator, critic, name, epoch):
    generator.save('models/%s_gan_generator_epoch_%d.h5' % (name, epoch))
    critic.save('models/%s_gan_critic_epoch_%d.h5' % (name, epoch))

def disp_sample_outputs(generator, critic, num_cols, num_rows, name, epoch, dlosses, glosses):
    fig = plt.figure(figsize=(10,10))
    rand_in = np.random.normal(0, 1, size=[num_cols*num_rows, gen_in_dim])
    print(generator.predict(rand_in))
    gen_out = generator.predict(rand_in)
    print("disc out:")
    print(critic.predict(gen_out))
    gen_out = gen_out.reshape(num_cols*num_rows, num_thresholded, 3)*[28, 28, 1]+[14, 14, 0]
    for i in range(1, num_cols*num_rows+1):
        fig.add_subplot(num_rows, num_cols, i)
        im_disp = np.zeros((28,28)) - 1.1
        for x in gen_out[i-1]:
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

#    plt.show()

def train_gan(generator, critic, gan, epochs, num_saves, name):
    gan_loss = []
    disc_loss = []
    # epochs_between_rows = int(float(epochs)/num_saves)

    y_real = np.ones(batch_size)
    y_gen = -np.ones(batch_size)
    for i in range(20, epochs):
        if(i%1==0):
            disp_sample_outputs(generator, critic, 10, 10, name, i, disc_loss, gan_loss)

        if(i%20==0):
            save_models(generator, critic, name, i)

        print("Epoch ", i)
        for _ in tqdm(range(batches_per_epoch)):
            dbatch_loss_real = 0
            dbatch_loss_fake = 0
            # Train critic
            critic.trainable = True

            for i in range(num_critic):
                noise = np.random.normal(0, 1, size=[batch_size, gen_in_dim])
                gen_batch = generator.predict(noise)
                image_batch = X_train[np.random.randint(0, X_train.shape[0], size=batch_size)]

                dbatch_loss_real += critic.train_on_batch(image_batch, y_real)
                dbatch_loss_fake += critic.train_on_batch(gen_batch, y_gen)

            for l in critic.layers:
                weights = l.get_weights()
                weights = [np.clip(w, -clip_value, clip_value) for w in weights]
                l.set_weights(weights)

            # Train critic
            # critic.trainable = True
            # dbatch_loss = critic.train_on_batch(X, np.concatenate([y_real, y_gen]))

            # Train generator
            noise = np.random.normal(0, 1, size=[batch_size, gen_in_dim])
            critic.trainable = False
            gbatch_loss = gan.train_on_batch(noise, y_real)

        # disc_loss.append(dbatch_loss)
        disc_loss.append((dbatch_loss_real+dbatch_loss_fake)/num_critic)
        gan_loss.append(gbatch_loss)

    disp_sample_outputs(generator, critic, 10, 10, name, epochs, disc_loss, gan_loss)
    save_models(generator, critic, name, epochs)

    return (disc_loss, gan_loss)

def run_trial(name, epochs):
    onlyfiles = [f for f in listdir('figs/') if isfile(join('figs/', f))]
    if (name + "_0.png" in onlyfiles):
        print("file name already used")
        # return
    name = name
    generator, critic, gan = init_models_pre_trained()
    generator.summary()
    critic.summary()
    gan.summary()
    disc_loss, gan_loss = train_gan(generator, critic, gan, epochs, epochs, name)
    plt.figure()
    plt.plot(disc_loss)
    # plt.ylim((0,10))
    plt.savefig("figs/" + name + "_disc_loss.png")
#    plt.show()

    plt.figure()
    plt.plot(gan_loss)
    # plt.ylim((0,10))
    plt.savefig("figs/" + name + "_gan_loss.png")
#    plt.show()

    print(disc_loss)
    print(gan_loss)

run_trial("1_test", 100)
