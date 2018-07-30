#%% 
from torch.utils.data.dataset import Dataset
from PIL import Image
from os import path
import pandas as pd
import numpy as np 
import torch
class ImageDataset(Dataset):
    def __init__(self, imgPath, listPath, transform=None):
        """Input:
            imgPath: il path alla cartella contenente le immagini
            listPath: il path al file contenente la lista delle immagini con le relative etichette
            transform: trasformazioni da applicare al dataset"""
        # conserviamo il path alla cartella contenente le immagini
        self.imgPath=imgPath
        # carichiamo la lista dei file
        self.images = np.loadtxt(listPath, dtype=str, delimiter=',')
        #It doesn't work self.images = pd.read_csv(listPath, sep=',', header=None)
        self.transform = transform
    
    def __getitem__(self, index):
        if len(self.images[index]) == 1:
            print(self.images[index])
            f = self.images[index]
            mode = 'test'
        else:    
            #recuperiamo pathName, x,y,u,v coordinate, l = etichetta classe 
            f,x,y,u,v,l = self.images[index]
            mode = 'training'
        # carichiamo l'immagine utilizzando PIL
        im = Image.open(path.join(self.imgPath, f))
        if self.transform is not None:
            im = self.transform(im).double()
        if mode == 'test':
            return {'image':im, 'path':f}
        label = int(l)
        x = float(x)
        y = float(y)
        u = float(u)
        v = float(v)
        target = np.array([x,y,u,v])
        return {'image':im, 'label':label, 'target':target}
    def __len__(self):
        return len(self.images)

class FeatureDataset(Dataset):
    def __init__(self, features, listPath):
        self.features = features
        self.list = np.loadtxt(listPath, dtype=str, delimiter=',')
    
    def __getitem__(self, index):
        #recuperiamo pathName, x,y,u,v coordinate, l = etichetta classe 
        f,x,y,u,v,l = self.list[index]
        x = float(x)
        y = float(y)
        u = float(u)
        v = float(v)
        target = np.array([x,y,u,v])
        features = torch.tensor(*self.features[index])
        return {'features': features, 'target': target}
    def __len__(self):
        return len(self.list)