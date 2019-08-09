from keras.models import load_model
import numpy as np

gen_in_dim = 100

model = load_model('models/6_normalized_ijgan_generator_epoch_30.h5')

rand_in = np.random.normal(0, 1, size=[1, gen_in_dim])
gen_out = model.predict(rand_in).reshape(1, 784, 3)

print(gen_out)
