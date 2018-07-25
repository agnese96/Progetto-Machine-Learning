#%%
import os
os.chdir('./Src')
path='..'
modelPath="C:/Users/beaut/Google Drive/Trio++/3°ANNO/Machine Learning/Progetto/Models/"
#%%
import torch 
import numpy as np 
from torch import nn
import time
from torch.optim import SGD
from torch.autograd import Variable
from sklearn.metrics import mean_absolute_error
from loadData import FeatureDataset, ImageDataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader



class NNRegressor(nn.Module):
    def __init__(self,in_features, out_features=4,hidden_units=200,):
        super(NNRegressor,self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features,hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units,hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units,out_features),
        )
    def forward(self,x):
        return self.model(x)

def trainRegression(model, train_loader, test_loader, lr=0.01, epochs=20, momentum=0.9, weight_decay = 0.000001):
    criterion = nn.MSELoss()
    optimizer = SGD(model.parameters(),lr, momentum=momentum)
    loaders = {'train':train_loader, 'validation':test_loader} 
    losses = {'train':[], 'validation':[]}
    accuracies = {'train':[], 'validation':[]}
    if torch.cuda.is_available(): 
        model=model.cuda()
    for e in range(epochs):
        for mode in ['train', 'validation']:
            if mode=='train': 
                model.train()
            else: 
                model.eval()
            epoch_loss = 0
            epoch_acc = 0
            samples = 0
            for i, batch in enumerate(loaders[mode]):
                input=Variable(batch["features"], requires_grad=True)
                target=Variable(batch["target"],)
                if torch.cuda.is_available(): 
                    input, target = input.cuda(), target.cuda(async=True)
                output = model(input)
                torch.cuda.synchronize()
                tm = time.time()
                l = criterion(output, target) 
                torch.cuda.synchronize()
                print('Loss: ',time.time()-tm)
                if mode=='train':
                    torch.cuda.synchronize()
                    tm = time.time()
                    l.backward()
                    torch.cuda.synchronize()
                    print('Backward: ',time.time()-tm)
                    optimizer.step()
                    optimizer.zero_grad()
                torch.cuda.synchronize()
                tm = time.time()
                acc = mean_absolute_error(target.data, output.data)
                print('Accuracy: ',time.time()-tm)
                epoch_loss+=l.item()*input.shape[0]
                epoch_acc+=acc*input.shape[0]
                samples+=input.shape[0]
                #print('Iteration: %d'%i)
            epoch_loss/=len(loaders[mode].dataset)
            epoch_acc/=len(loaders[mode].dataset)
            losses[mode].append(epoch_loss)
            accuracies[mode].append(epoch_acc)
            print ("\r[%s] Epoch %d/%d. Iteration %d/%d. Loss: %0.2f. Accuracy: %0.2f\t\t\t\t\t" % \
            (mode, e+1, epochs, i, len(loaders[mode]), epoch_loss, epoch_acc),)
    #restituiamo il modello e i vari log
    return model, (losses, accuracies) 
#%%
#%% 
CNNOutputTrain = torch.load(modelPath+'FeaturesResNet18Train512.pth')
CNNOutputValidation = torch.load(modelPath+'FeaturesResNet18Validation512.pth')

#%%
TrainDataset = FeatureDataset(CNNOutputTrain, path+'/Dataset/training_list.csv')
ValidationDataset = FeatureDataset(CNNOutputValidation, path+'/Dataset/validation_list.csv')

NNRegressorModel = NNRegressor(512,4)

#%% definiamo i data loaders
featureLoaderTrain = DataLoader(TrainDataset, batch_size=5, num_workers=0, shuffle=True)
featureLoaderValidation = DataLoader(ValidationDataset, batch_size=5, num_workers=0)
