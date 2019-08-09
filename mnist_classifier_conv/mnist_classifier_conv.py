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
from keras.layers import Dense, LeakyReLU, Dropout, Input, Conv1D, MaxPool1D, Flatten
from keras.optimizers import Adam, RMSprop
from keras import initializers
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.cm as cm
from keras.utils import to_categorical
# import os

from os import listdir
from os.path import isfile, join
from tqdm import tqdm
# from mbdiscriminator import MinibatchDiscrimination
#
# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# K.set_session(sess)
#
# print(K.tensorflow_backend._get_available_gpus())
#
K.set_image_dim_ordering('th')

np.random.seed(1000)
num_thresholded = 100

(X_train_pre, Y_train), (X_test_pre, Y_test) = mnist.load_data()

#converting to single vector and normalized
X_train_pre = (X_train_pre.astype(np.float32)-127.5)/255

X_train = []

imrange = np.linspace(-0.5, 0.5, num=28, endpoint=False)

xs, ys = np.meshgrid(imrange, imrange)

xs = xs.reshape(-1)
ys = ys.reshape(-1)

X_train=np.array(list(map(lambda x: np.array([xs, ys, x.reshape(-1)]).T, X_train_pre)))
X_train=np.array(list(map(lambda x: x[x[:,2].argsort()][-num_thresholded:], X_train)))


X_test_pre = (X_test_pre.astype(np.float32)-127.5)/255

X_test = []

imrange = np.linspace(-0.5, 0.5, num=28, endpoint=False)

xs, ys = np.meshgrid(imrange, imrange)

xs = xs.reshape(-1)
ys = ys.reshape(-1)

X_test=np.array(list(map(lambda x: np.array([xs, ys, x.reshape(-1)]).T, X_test_pre)))
X_test=np.array(list(map(lambda x: x[x[:,2].argsort()][-num_thresholded:], X_test)))

Y_train = to_categorical(Y_train, 10)
Y_test = to_categorical(Y_test, 10)

print(X_train.shape)

batch_size = 128
batches_per_epoch = int(X_train.shape[0]/batch_size)
learning_rate = 0.0002
disc_dropout = 0.3
clip_value = 0.01
num_critic = 5

def init_model():

    adam = Adam(lr=learning_rate, beta_1 = 0.5)
    rmsprop = RMSprop(lr=learning_rate)

    optim = adam

    classifier = Sequential()

    classifier.add(Conv1D(32, kernel_size=3, input_shape=(num_thresholded,3)))
    classifier.add(LeakyReLU(alpha=0.2))
    # classifier.add(Dropout(0.3))

    classifier.add(Conv1D(64, kernel_size=3))
    classifier.add(LeakyReLU(alpha=0.2))

    classifier.add(MaxPool1D(pool_size=2))
    classifier.add(Dropout(0.3))

    classifier.add(Flatten())

    classifier.add(Dense(128))
    classifier.add(LeakyReLU(alpha=0.2))
    classifier.add(Dropout(0.5))

    classifier.add(Dense(10, activation='softmax'))

    classifier.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])

    return classifier

def save_models(classifier, name, epoch):
    classifier.save('models/%s_epoch_%d.h5' % (name, epoch))

classifier = init_model()
classifier.summary()

classifier.fit(X_train, Y_train, batch_size=batch_size, epochs=100, verbose=1, validation_data=(X_test, Y_test))
