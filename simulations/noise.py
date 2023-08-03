import numpy as np
from skimage import data, img_as_float, io, util, img_as_ubyte
from matplotlib.image import imsave
import os
import random


# # Importer l'image de la camÃ©ra et la convertir en format float
# image = img_as_float(data.simulation())
# image_int = np.array(image*255).astype('uint8')
# # Enregistrer l'image originale
# image_destination = "noised_images/camera.png"
# io.imsave(image_destination, image_int)

for i in range(100):
    nom = f'simulations_clean_train/simulation{i}.png'
    image = io.imread(nom)


    var = 0.01*random.uniform(1,3)
    noisy_image = util.random_noise(image, mode='gaussian',var=var)

    # Enregistrer l'image bruitÃ©e avec le bruit de Poisson
    noisy = np.array(noisy_image*255).astype('uint8')
    image_destination = "gaussian_images/simulation_gaussian"+str(i)+".png"
    io.imsave(image_destination, noisy)



    # La somme de deux variables alÃ©toires suivant un loi de poisson suit un loi de poisson de paramÃ©tre la somme des deux paramÃ©tres.
    noisy_image = image

    noisy_image = util.random_noise(noisy_image, mode='poisson')

    noisy = np.array(noisy_image*255).astype('uint8')
    image_destination = "poisson_images/simulation_poisson"+str(i)+".png"
    io.imsave(image_destination, noisy)
