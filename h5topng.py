import h5py
from PIL import Image
from sys import argv
import numpy as np
from skimage import img_as_float
from skimage import io
import glob
import os
import numpy as np



directory = argv[1]  

max_global = float('-inf')
min_global = float('inf')

# Parcourir tous les fichiers h5 dans le répertoire
for file_path in glob.glob(directory + '/*'):
    with h5py.File(file_path, 'r') as f:
        # Obtenez le nom du dataset que vous souhaitez convertir
        dataset_name = 'dat'
        # Accédez au dataset
        dataset = f[dataset_name]
        # Convertissez le dataset en un tableau numpy avec le type de données 'uint8'
        data = np.array(dataset)

  
    max_val = np.max(data)
    min_val = np.min(data)
    max_global = max(max_global, max_val)
    min_global = min(min_global, min_val)

min_global = max(0.0,min_global)

# Parcourir à nouveau les fichiers pour créer les images
for file_path in glob.glob(directory + '/*'):
    with h5py.File(file_path, 'r') as f:
        dataset_name = 'dat'
        dataset = f[dataset_name]
        data = np.array(dataset)

    max_int = ((2**32) - 1)
    image = (data - min_global) * (max_int / (max_global - min_global))
    image = np.array(image).astype("uint32")

    path = argv[2] 
    io.imsave(f"{path}/{os.path.basename(file_path)}"[:-3] + ".png", image)



