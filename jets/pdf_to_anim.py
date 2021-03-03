import pdf2image
from os import listdir
import cv2
import numpy as np


figdir = '/graphganvol/mnist_graph_gan/jets/figs/66_g30/'

fnames = listdir(figdir)

print(fnames)

# fnames = fnames[1:]

images = []
for fname in fnames:
    fullfname = figdir + fname
    images.append(pdf2image.convert_from_path(fullfname))

videodims = (5751, 1339)
fourcc = cv2.VideoWriter_fourcc(*'avc1')
video = cv2.VideoWriter(figdir + "anim.mp4", fourcc, 60, videodims)

for im in images:
    video.write(np.array(im[0]))

video.release()
