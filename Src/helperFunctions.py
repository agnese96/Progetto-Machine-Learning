#%%
import numpy as np
import torch
from matplotlib import pyplot as plt

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

#%% plottiamo i log
def plot_logs_classification(logs):
    train_losses, train_acc, test_losses, test_acc = \
        logs[0]['train'], logs[1]['train'], logs[0]['validation'], logs[1]['validation']
    plt.figure(figsize=(18,6))
    plt.subplot(121)
    plt.plot(train_losses)
    plt.plot(test_losses)
    plt.legend(['Training loss', 'Testing loss'])
    plt.grid()
    plt.subplot(122)
    plt.plot(train_acc)
    plt.plot(test_acc)
    plt.legend(['Training accuracy', 'Testing accuracy'])
    plt.grid()
    plt.show()