#%% 
from torch.utils.data.dataset import Dataset
from PIL import Image
from os import path
import pandas as pd

class ImageDataset(Dataset):
    def __init__(self, imgPath, listPath, transform=None):
        """Input:
            imgPath: il path alla cartella contenente le immagini
            listPath: il path al file contenente la lista delle immagini con le relative etichette
            transform: trasformazioni da applicare al dataset"""
        # conserviamo il path alla cartella contenente le immagini
        self.imgPath=imgPath
        # carichiamo la lista dei file
        self.images = pd.read_csv(listPath, sep=',', header=None)
        self.transform = transform
    
    def __getitem__(self, index):
        #recuperiamo il path dell'immagine di indice index e la relativa etichetta
        f = self.images[index][0]
        # carichiamo l'immagine utilizzando PIL
        im = Image.open(path.join(self.imgPath, f))
        if self.transform is not None:
            im = self.transform(im)
        return {'image':im, 'features':self.images[index]}
    def __len__(self):
        return len(self.images)