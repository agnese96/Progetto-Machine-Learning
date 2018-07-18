#%%
import numpy as np
import torch

#%%
from torch.autograd import Variable

def mediaVarianzaImmagini(dataset):
  m = np.zeros(3)
  s = np.zeros(3)
  
  # Calcoliamo la la media:    
  for sample in dataset:
      m+=sample['image'].sum(1).sum(1) # accumuliamo la somma dei pixel canale per canale
  # dividiamo per il numero di immagini moltiplicato per il numero di pixel
  m = m/(len(dataset)*144*256)

  # Calcoliamo la deviazione standard:
  for sample in dataset:
    s+=((sample['image']-torch.Tensor(m).view(3,1,1))**2).sum(1).sum(1)
  s = np.sqrt(s/len(dataset)*144*256)
  
  return m,s