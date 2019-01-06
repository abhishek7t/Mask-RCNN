import numpy as np 
import pdb
import tensorflow as tf

import os

from PIL import Image, ImageOps
import numpy as np
import scipy.misc
from scipy.misc import imread
from matplotlib import pyplot as plt

import spatial_transformer
# theta[0,0] = w/128
# theta[0,1] = 0
# theta[0,2] = (x-64)/64
# theta[0,3] = 0
path = "/home/abhishek/Downloads/CIS680_2017Fall-master/hw2.b/P&C dataset"
w = 20
h = 51
x = 3 + w/2.0
y = 54 + h/2.0

theta = np.array([w/128.0, 0.0, (x-64)/64.0, 0, h/128.0, (y-64)/64.0]) 
image_path = os.path.join(path, "img", "000001.jpg")
img = imread(image_path)
out_img = spatial_transformer.transformer(img.reshape([1,128,128,3]), theta.reshape([1,6]), out_size=(22,22))
with tf.Session() as sess:
    out = sess.run(out_img)
    # pdb.set_trace()
    image = out[0]
    image = np.clip(image, 0, 255).astype('uint8')
    scipy.misc.imsave('transformed_image.png', image)
    plt.imshow(out[0].astype(np.uint8))
    plt.show()
pdb.set_trace()
print(theta)