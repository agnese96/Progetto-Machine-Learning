#%% 
from torch.utils.data.dataset import Dataset
from PIL import Image
from os import path
import pandas as pd
import numpy as np 
import torch
#%%
class ImageDataset(Dataset):
    def __init__(self, imgPath, listPath, transform=None, mode='training'):
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
        self.mode = mode
    
    def __getitem__(self, index):
        if self.mode=='test':
            f = self.images[index]
        else:    
            #recuperiamo pathName, x,y,u,v coordinate, l = etichetta classe 
            f,x,y,u,v,l = self.images[index]
        # carichiamo l'immagine utilizzando PIL
        im = Image.open(path.join(self.imgPath, f))
        if self.transform is not None:
            im = self.transform(im)
        if self.mode == 'test':
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
        if listPath is not None:
            self.list = np.loadtxt(listPath, dtype=str, delimiter=',')
            self.mode = 'training'
        else:
            self.mode = 'test'
    
    def __getitem__(self, index):
        features = torch.tensor(*self.features[index])
        if self.mode == 'test':
            return {'features': features}
        #recuperiamo pathName, x,y,u,v coordinate, l = etichetta classe 
        f,x,y,u,v,l = self.list[index]
        x = float(x)
        y = float(y)
        u = float(u)
        v = float(v)
        target = np.array([x,y,u,v])
        return {'features': features, 'target': target}
    def __len__(self):
        return len(self.list)