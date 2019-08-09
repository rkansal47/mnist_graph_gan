import matplotlib.pyplot as plt
import numpy as np

dataset = np.loadtxt('../mnist_dataset/mnist8m.csv', delimiter=',', dtype=np.float32)

plt.imshow(dataset[0])
plt.save("test_im.png")
