import numpy as np
from skimage import img_as_float
import matplotlib.pyplot as plt
from skimage import io
import random 
from sys import argv


def generate(data,indice):
    matrice = []
    for i in range(256):
        ligne = []
        for j in range(256):
            pixelij = 0
            for tpl in data :
                # On enlève ce terme : (1/np.sqrt(2*np.pi*tpl[1])) 
                val_max = 1
                val_pixel = np.exp(-((i-tpl[0][0])**2 + (j-tpl[0][1])**2)/(2*tpl[1]))
                if val_pixel >= tpl[3]*val_max:
                    pixelij += tpl[3]*val_max
                else :
                    pixelij += val_pixel
            ligne.append(pixelij)
        matrice.append(ligne)

    matrice = np.array(matrice)
    img = img_as_float(matrice)
    img = np.array(img*255).astype('uint8')
    plt.imshow(img[:,:,...])
    # plt.title('Input')
    # plt.show()
    nom = argv[1]+f'/simulation{indice}.png'
    io.imsave(nom, img)



if __name__=='__main__':
    # data = [moyenne, variance, intensity,pourcentage]
    for i in range (100):
        data = [([random.uniform(0,255),random.uniform(0,255)],random.uniform(50,200),random.uniform(2,3),random.uniform(0.4,0.8)),
                 ([random.uniform(0,255),random.uniform(0,255)],random.uniform(50,200),random.uniform(2,3),random.uniform(0.4,0.8)),
                 ([random.uniform(0,255),random.uniform(0,255)],random.uniform(50,200),random.uniform(2,3),random.uniform(0.4,0.8)),
                 ([random.uniform(0,255),random.uniform(0,255)],random.uniform(50,200),random.uniform(2,3),random.uniform(0.4,0.8)),
                 ([random.uniform(0,255),random.uniform(0,255)],random.uniform(50,200),random.uniform(2,3),random.uniform(0.4,0.8)),
        ]
        generate(data,i)
