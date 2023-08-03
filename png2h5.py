import numpy as np
import h5py
from PIL import Image
from sys import argv
import glob
import os

directory = argv[1]  #Directory of images to convert

for image_path in glob.glob(directory + '/*'):

    # Charger l'image PNG en utilisant PIL
    image = Image.open(image_path)
    

    image = image.convert('L')

    # Convertir l'image en un tableau NumPy
    image_array = np.array(image)

    # Créer le fichier HDF5
    output_file = os.path.basename(image_path)
    output_file = argv[2] + output_file[:-4] + ".h5"
    with h5py.File(output_file, 'w') as h5_file:
        # Créer un jeu de données dans le fichier HDF5
        h5_file.create_dataset('dat', data=image_array)

    print("Conversion terminée. Fichier HDF5 créé :", output_file)
