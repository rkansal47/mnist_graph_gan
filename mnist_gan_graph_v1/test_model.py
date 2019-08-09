import setGPU
# from keras.datasets import mnist
# from keras.models import Sequential, Model
# from keras.layers import Dense, LeakyReLU, Dropout, Input
# from keras.optimizers import Adam
# from keras import initializers
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
# import matplotlib.cm as cm
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
from keras.models import load_model

def wasserstein_loss(y_true, y_pred):
    return -K.mean(y_true*y_pred)

disc = load_model('models/14_1sgan_discriminator_epoch_200.h5', custom_objects={'wasserstein_loss': wasserstein_loss})
gen = load_model('models/14_1sgan_generator_epoch_200.h5', custom_objects={'wasserstein_loss': wasserstein_loss})

noise = np.random.normal(0, 1, size=[100, 100])
gen_out = gen.predict(noise)

# print(gen_out)

print("")

disc_out = disc.predict(gen_out)
print(disc_out)
