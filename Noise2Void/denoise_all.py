import os
from sys import argv

# Chemin du répertoire contenant les fichiers
repertoire = argv[1]  # path to images to denoise

# Commandes à exécuter
commande = 'python3 prediction.py'

# Parcourir tous les fichiers du répertoire
for nom_fichier in os.listdir(repertoire):
    chemin_fichier = os.path.join(repertoire, nom_fichier)
    
    # Exécuter la première commande sur le fichier
    model_name = argv[2]
    os.system(f'{commande} {chemin_fichier} {model_name}')