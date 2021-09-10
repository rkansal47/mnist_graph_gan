#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
file: lagan.py
description: submodels for [arXiv/1701.05927]
author: Luke de Oliveira (lukedeoliveira@lbl.gov)
"""

import keras.backend as K
from keras.layers import (Input, Dense, Reshape, Flatten, Lambda, Concatenate, Multiply,
                          Dropout, BatchNormalization, Activation, Embedding)
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import (UpSampling2D, Conv2D, ZeroPadding2D,
                                        AveragePooling2D)
from keras.layers.local import LocallyConnected2D

from keras.models import Model, Sequential

from ops import minibatch_discriminator, minibatch_output_shape, Dense3D


K.set_image_data_format('channels_last')


def discriminator():

    image = Input(shape=(25, 25, 1))

    # block 1: normal 5x5 conv,
    # *NO* batchnorm (recommendation from [arXiv/1511.06434])
    x = Conv2D(32, 5, padding='same')(image)
    x = LeakyReLU()(x)
    x = Dropout(0.5)(x)

    # block 2: 'same' bordered 5x5 locally connected block with batchnorm and
    # 2x2 subsampling
    x = ZeroPadding2D((2, 2))(x)
    x = LocallyConnected2D(8, 5, strides=2, padding='valid')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    # block 2: 'same' bordered 5x5 locally connected block with batchnorm
    x = ZeroPadding2D((2, 2))(x)
    x = LocallyConnected2D(8, 5, padding='valid')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    # block 3: 'same' bordered 3x3 locally connected block with batchnorm and
    # 2x2 subsampling
    x = ZeroPadding2D((1, 1))(x)
    x = LocallyConnected2D(8, 3, strides=2, padding='valid')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    x = AveragePooling2D((2, 2))(x)
    h = Flatten()(x)

    dnn = Model(image, h)  # output

    image = Input(shape=(25, 25, 1))

    dnn_out = dnn(image)

    # # nb of features to obtain
    # nb_features = 20
    #
    # # dim of kernel space
    # vspace_dim = 10
    #
    # # creates the kernel space for the minibatch discrimination
    # K_x = Dense3D(nb_features, vspace_dim)(dnn_out)
    #
    # minibatch_featurizer = Lambda(minibatch_discriminator, output_shape=minibatch_output_shape)
    #
    # # print('test')
    #
    # # concat the minibatch features with the normal ones
    # features = Concatenate(axis=-1)([minibatch_featurizer(K_x), dnn_out])

    # fake output tracks binary fake / not-fake, and the auxiliary requires
    # reconstruction of latent features, in this case, labels
    fake = Dense(1, activation='sigmoid', name='generation')(dnn_out)
    # aux = Dense(1, activation='sigmoid', name='auxiliary')(features)

    return Model(image, fake)


def generator(latent_size, return_intermediate=False):

    loc = Sequential([
        # DCGAN-style project & reshape,
        Dense(128 * 7 * 7, input_dim=latent_size),
        Reshape((7, 7, 128)),

        # block 1: (None, 7, 7, 128) => (None, 14, 14, 64),
        Conv2D(64, 5, padding='same', kernel_initializer='he_uniform'),
        LeakyReLU(),
        BatchNormalization(),
        UpSampling2D(size=(2, 2)),

        # block 2: (None, 14, 14, 64) => (None, 28, 28, 6),
        ZeroPadding2D((2, 2)),
        LocallyConnected2D(6, 5, kernel_initializer='he_uniform'),
        LeakyReLU(),
        BatchNormalization(),
        UpSampling2D(size=(2, 2)),

        # block 3: (None, 28, 28, 6) => (None, 25, 25, 1),
        LocallyConnected2D(6, 3, kernel_initializer='he_uniform'),
        LeakyReLU(),
        LocallyConnected2D(1, 2, use_bias=False, kernel_initializer='glorot_normal', activation='tanh'),
        # Activation('sigmoid')
    ])

    # this is the z space commonly refered to in GAN papers
    latent = Input(shape=(latent_size, ))

    # # this will be our label
    # image_class = Input(shape=(1, ), dtype='int32')
    # emb = Flatten()(Embedding(2, latent_size, input_length=1, embeddings_initializer='glorot_normal')(image_class))
    #
    # # hadamard product between z-space and a class conditional embedding
    # h = Multiply()([latent, emb])

    fake_image = loc(latent)

    return Model(latent, fake_image)
