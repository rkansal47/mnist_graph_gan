# import setGPU
# import tensorflow as tf

# with tf.Session() as sess:
# 	devices = sess.list_devices()
# 	print("devices")
# 	print(devices)
#
# print("GPU device")
# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))


import setGPU
from keras import backend as K
K.tensorflow_backend._get_available_gpus()
