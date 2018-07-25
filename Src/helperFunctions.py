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

#%% plottiamo i log
def plot_logs_regression(logs):
    train_losses, train_acc, test_losses, test_acc = \
        logs[0]['train'], logs[1]['train'], logs[0]['validation'], logs[1]['validation']
    plt.figure(figsize=(18,6))
    plt.subplot(121)
    plt.plot(train_losses)
    plt.plot(test_losses)
    plt.yscale('log')
    plt.legend(['Training loss', 'Testing loss'])
    plt.grid()
    plt.subplot(122)
    plt.plot(train_acc)
    plt.plot(test_acc)
    plt.yscale('log')
    plt.legend(['Training accuracy', 'Testing accuracy'])
    plt.grid()
    plt.show()

#MODO 2) CURVA ROC 
#Calcoliamo tutti gli errori tra il valore F(xi) e yi quello predetto. Poi li raggruppiamo in 'bin' e li cumuliamo rappresentandoli via via. 
def rec_curve(predictions, reals):
    #calcoliamo gli errori ovvero Ei =(f(xi) - yi)^2
    errors = (predictions - reals)**2
    #ogni "errore " sarà un bin. Ovviamente dobbiamo evitare bin duplicaati: . Dopodichè ordiniamo dal più piccolo errore al più grande.
    #Ci aspettiamo che la maggior degli errori cada nei primi bin, cioè quelli più piccoli. Ogni bin sarà un valore 'x' nell'asse x. 
    bins =np.sort( np.unique(errors) )
    #troviamo le frequenze 'cumulate' rispetto ad ogni bin. Ogni elemento sarà un punto 'y' rispetto ad un bin 'x'. 
    freq=[]
    for bin in bins: 
        numElement = (errors <=bin).mean() #trova gli elementi <= all'errore rappresentato dal bin e fanne la media. Usiamo 'mean' perchè 
        #vogliamo trovare la frazione rispetto al totale elementi( con sum trovavamo il numero di quelli solo <=bin)
        freq.append( numElement)

    AUC = np.trapz(freq,bins) #calcoliamo l'area sotto la curva ottenuta considerando i bin sull'asse x e le frequenze 'freq' sull'asse y
    #l'area totale è un rettangolo la cui base è pari al bin più grande sull'asse x mentre l'altezza è pari ad 1, perchè stiamo considerando 
    #solo frazioni sull'asse y quindi valori compresi tra 0 e 1 ( max. ). Area rettangolo è base per altezza-> lastBin*1
    #Quindi l'area totale meno quella sotto la curva, ci da quella sopra la curva che deve essere più piccola possibile 
    totArea = np.max(bins)*1  #area totale
    AOC = totArea - AUC #area sopra la curva . Deve esserre più piccola possibile
    #---------vog
    return bins,freq,AOC #restituisce tutto come una tupla. quindi un array sostanzialmente  
    
#boston_linear_regressor_rec =rec_curve(boston_linear_regressor_predictions,boston_linear_regressor_reals)


