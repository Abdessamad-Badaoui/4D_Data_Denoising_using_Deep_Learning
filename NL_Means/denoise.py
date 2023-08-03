import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread, imsave
from skimage import data, img_as_float
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.util import random_noise
from sys import argv
import os
from matplotlib.image import imsave
import time

def PSNR(image1, image2):
    mse = np.mean((image1 - image2) ** 2)
    max_pixel_value = np.max(image1)
    psnr = 20 * np.log10(max_pixel_value) - 10 * np.log10(mse)
    return psnr


def bsd3_denoising(image_path):
    # Charger l'image d'entrée

    img = img_as_float(imread(image_path))
    

    if img.ndim == 2:
        img = np.stack((img,)*3,axis=-1)

    # estimate the noise standard deviation from the noisy image
    sigma_est = np.mean(estimate_sigma(img, channel_axis=-1))

    patch_kw = dict(patch_size=5,      # 5x5 patches
                patch_distance=6,  # 13x13 search area
                channel_axis=-1)


    # Appliquer l'algorithme BSD3 pour le débruitage
    print(img)
    start_time = time.time()
    denoised = denoise_nl_means(img, h=1.15 * sigma_est, fast_mode=False,
                                **patch_kw)
    end_time = time.time()
    print("Temps = ",end_time - start_time)

   

    # image_name = os.path.basename(argv[1])
    # output_path = "denoised_images/denoised_" + image_name
    # imsave(output_path, denoised)
    # print("L'image débruitée a été enregistrée sous", output_path)

    # We show the noisy input...
    plt.subplot(1,2,1)
    plt.imshow( img[:,:,...] )
    plt.title('Input')

    # and the result.
    plt.subplot(1,2,2)
    plt.imshow( denoised[:,:,...] )
    plt.title('Prediction')
    plt.show()
    
    # Importer l'image GT:
    imgGT = imread(argv[2])
    print("Le PSNR :", PSNR(denoised[:,:,0],imgGT))

    image_name = os.path.basename(argv[1])
    image_destination = "denoised_images/pred_" + image_name
    imsave(image_destination, np.clip(denoised,0.0,1.0),cmap='gray')


if __name__=='__main__':
    # Récupérer le chemin de l'image depuis les arguments du terminal
    image_path = argv[1]

    # Appeler la fonction de débruitage BSD3 
    bsd3_denoising(image_path)





