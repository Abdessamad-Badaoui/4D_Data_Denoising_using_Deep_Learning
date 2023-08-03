import os
import cv2
from sys import argv

# Chemin du répertoire contenant les images en noir et blanc
repertoire_entree = argv[1]

# Chemin du répertoire pour enregistrer les images en couleurs
repertoire_sortie = argv[2]

# Liste des fichiers dans le répertoire d'entrée
fichiers = os.listdir(repertoire_entree)

# Parcourir tous les fichiers du répertoire d'entrée
for fichier in fichiers:
    chemin_fichier_entree = os.path.join(repertoire_entree, fichier)
    
    # Charger l'image en noir et blanc
    image_noir_et_blanc = cv2.imread(chemin_fichier_entree, 0)
    
    # Appliquer la colormap "jet"
    image_en_couleurs = cv2.applyColorMap(image_noir_et_blanc, cv2.COLORMAP_JET)
    
    # Chemin du fichier de sortie
    chemin_fichier_sortie = os.path.join(repertoire_sortie, fichier)
    
    # Enregistrer l'image en couleurs
    cv2.imwrite(chemin_fichier_sortie, image_en_couleurs)