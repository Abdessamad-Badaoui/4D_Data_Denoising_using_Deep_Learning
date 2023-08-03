# We import all our dependencies.
from n2v.models import N2V
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.image import imread, imsave
from csbdeep.io import save_tiff_imagej_compatible
from sys import argv
import os


# A previously trained model is loaded by creating a new N2V-object without providing a 'config'.  
# model_name = n2v_2D_gaussian par exemple
model_name = argv[2]
basedir = 'models'
model = N2V(config=None, name=model_name, basedir=basedir)

# We read the image we want to process and get rid of the Alpha channel.
img = imread(argv[1])

# Here we process the image.
pred = model.predict(img, axes='YX',tta=True)

image_name = os.path.basename(argv[1])
image_destination = "denoised_images/pred_" + image_name
from matplotlib.image import imsave
imsave(image_destination, np.clip(pred,0.0,1.0),cmap='gray')