import pdf2image
import cv2
import numpy as np
from tqdm import tqdm


figdir = '/graphganvol/mnist_graph_gan/jets/figs/150_g30_anim/'

images = []
for i in tqdm(range(150)):
    fullfname = figdir + str(i) + 'p.pdf'
    images.append(pdf2image.convert_from_path(fullfname))

videodims = (4147, 1340)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(figdir + "anim.mp4", fourcc, 10, videodims)

for i in tqdm(range(len(images))):
    video.write(np.array(images[i][0]))

video.release()
