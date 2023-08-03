import sys
sys.path.append("..")
import torch
from skimage import io, color
from matplotlib.image import imsave, imread
from util import show, plot_images, plot_tensors
from matplotlib import pyplot as plt
from skimage import data, img_as_float
import pickle
import numpy as np
from sys import argv
import os
from models.dncnn import DnCNN

new_model = DnCNN(1, num_of_layers = 18)
# argv[3] = best_weights_...
loaded_model = new_model.load_weights(argv[3], num_of_layers = 18)

# We read the image we want to process and get rid of the Alpha channel.
img = io.imread(argv[1], as_gray=True)
img = img_as_float(img)
noisy_image = np.array(img).astype(float)
noisy = torch.Tensor(noisy_image[np.newaxis, np.newaxis])

# Here we process the image.
denoised = np.clip(loaded_model(noisy).detach().cpu().numpy()[0, 0], 0, 1).astype(np.float64)

# Let's look at the results.
plt.figure(figsize=(30,30))

# We show the noisy input...
plt.subplot(1,2,1)
plt.imshow( img,cmap='gray' )
plt.title('Input')

# and the result.
plt.subplot(1,2,2)
plt.imshow( denoised,cmap='gray' )
plt.title('Prediction')
plt.show()

def PSNR(image1, image2):
    mse = np.mean((image1 - image2) ** 2)
    max_pixel_value = np.max(image1)
    psnr = 20 * np.log10(max_pixel_value) - 10 * np.log10(mse)
    return psnr

# Importer l'image GT:
imgGT = imread(argv[2])
print("Le PSNR :", PSNR(imgGT,denoised))

image_name = os.path.basename(argv[1])
image_destination = "denoised_images/pred_" + image_name
from matplotlib.image import imsave
imsave(image_destination, np.clip(denoised,0.0,1.0),cmap='gray')
