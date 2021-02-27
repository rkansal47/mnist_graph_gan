import pdf2image
from os import listdir

images = pdf2image.convert_from_path('figs/test/0j.pdf')
images


figdir = 'figs/test/'

fnames = listdir(figdir)

fnames

fnames = fnames[1:]

images = []
for fname in fnames:
    fullfname = figdir + fname
    images.append(pdf2image.convert_from_path(fullfname))

images

from PIL import Image, ImageDraw
import cv2

videodims = (1744, 406)
fourcc = cv2.VideoWriter_fourcc(*'avc1')
video = cv2.VideoWriter("test.mp4", fourcc, 60, videodims)

img = Image.new('RGB', videodims, color = 'darkred')

for i in range(len(images)):
    imtemp = img.copy()
    # draw frame specific stuff here.
    video.write(cv2.cvtColor(np.array(imtemp), cv2.COLOR_RGB2BGR))

video.release()
