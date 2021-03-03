import pdf2image
import cv2
import numpy as np
from tqdm import tqdm


figdir = '/graphganvol/mnist_graph_gan/jets/figs/150_g30_anim/'

images = []
for i in tqdm(range(151)):
    fullfname = figdir + str(i) + 'p.pdf'
    images.append(pdf2image.convert_from_path(fullfname))

videodims = (5751, 1339)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(figdir + "anim.mp4", fourcc, 60, videodims)

for i in tqdm(range(len(images))):
    video.write(np.array(images[i][0]))

video.release()
