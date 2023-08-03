import numpy as np
from skimage import io, img_as_float, img_as_ubyte
from skimage.restoration import denoise_wavelet
from sys import argv
from matplotlib.image import imsave
import os
import matplotlib.pyplot as plt
from matplotlib.image import imread

def PSNR(image1, image2):
    mse = np.mean((image1 - image2) ** 2)
    max_pixel_value = np.max(image1)
    psnr = 20 * np.log10(max_pixel_value) - 10 * np.log10(mse)
    return psnr


def bm3d_denoising(image_path):
    # Charger l'image en tant que flottant
    noisy_image = img_as_float(io.imread(image_path))

    # Débruitage avec BM3D
    denoised_image = denoise_wavelet(noisy_image)

    # We show the noisy input...
    plt.subplot(1,2,1)
    plt.imshow( noisy_image[:,:,...] )
    plt.title('Input')

    # and the result.
    plt.subplot(1,2,2)
    plt.imshow( denoised_image[:,:,...] )
    plt.title('Prediction')
    plt.show()

    imgGT = imread(argv[2])
    print("Le PSNR :", PSNR(denoised_image[:,:],imgGT))

    image_name = os.path.basename(image_path)
    image_destination = "denoised_images/pred_" + image_name
    imsave(image_destination, np.clip(denoised_image,0.0,1.0),cmap='gray')



if __name__=="__main__":
    # Exemple d'utilisation
    image_path = argv[1]
    bm3d_denoising(image_path)