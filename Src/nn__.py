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
            nn.Linear(in_features,400),
            nn.ReLU(),
            nn.Linear(400,300),
            nn.ReLU(),
            nn.Linear(300,200),
            nn.ReLU(),
            nn.Linear(200,100),
            nn.ReLU(),
            nn.Linear(100,100),
            nn.ReLU(),
            nn.Linear(100,50),
            nn.ReLU(),
            nn.Linear(50,out_features),
        )
    def forward(self,x):
        return self.model(x)
#%%
def localizationLoss(input, target, beta=0.7):
    x_pred = input[:,0]
    y_pred = input[:,1]
    u_pred = input[:,2]
    v_pred = input[:,3]
    x = target[:,0]
    y = target[:,1]
    u = target[:,2]
    v = target[:,3]
    return torch.mean((x_pred-x)**2+(y_pred-y)**2 + beta*((u_pred-u)**2+(v_pred-v)**2))

def trainRegression(model, train_loader, test_loader, lr=0.001, epochs=20, momentum=0.8, weight_decay = 0.000001):
    criterion = localizationLoss
    optimizer = SGD(model.parameters(),lr, momentum=momentum, weight_decay=weight_decay)
    loaders = {'train':train_loader, 'validation':test_loader} 
    losses = {'train':[], 'validation':[]}
    mse_cumulative = {'train':[], 'validation':[]}
    if torch.cuda.is_available(): 
        model=model.cuda()
    for e in range(epochs):
        for mode in ['train', 'validation']:
            if mode=='train': 
                model.train()
            else: 
                model.eval()
            epoch_loss = 0
            epoch_mse = 0
            samples = 0
            for i, batch in enumerate(loaders[mode]):
                input=Variable(batch["features"], requires_grad=True)
                target=Variable(batch["target"],)
                if torch.cuda.is_available(): 
                    input, target = input.cuda(), target.cuda(async=True)
                output = model(input).squeeze(1)
                l = criterion(output, target)
                if mode=='train':
                    l.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                mse = mean_absolute_error(target.data, output.data)
                epoch_loss+=l.item()*input.shape[0]
                epoch_mse+=mse*input.shape[0]
                samples+=input.shape[0]
                #print('Iteration: %d'%i)
            epoch_loss/=len(loaders[mode].dataset)
            epoch_mse/=len(loaders[mode].dataset)
            losses[mode].append(epoch_loss)
            mse_cumulative[mode].append(epoch_mse)
            print ("\r[%s] Epoch %d/%d. Iteration %d/%d. Loss: %0.2f. Mean Absolute Error: %0.2f\t\t\t\t\t" % \
            (mode, e+1, epochs, i, len(loaders[mode]), epoch_loss, epoch_mse),)
    #restituiamo il modello e i vari log
    return model, (losses, mse_cumulative) 

#%% 
CNNOutputTrain = torch.load(modelPath+'FeaturesResNet18Train512.pth')
CNNOutputValidation = torch.load(modelPath+'FeaturesResNet18Validation512.pth')

#%%
TrainDataset = FeatureDataset(CNNOutputTrain, path+'/Dataset/training_list.csv')
ValidationDataset = FeatureDataset(CNNOutputValidation, path+'/Dataset/validation_list.csv')

NNRegressorModel = NNRegressor(512,4)
NNRegressorModel.double()
#%% definiamo i data loaders
featureLoaderTrain = DataLoader(TrainDataset, batch_size=200, num_workers=0, shuffle=True)
featureLoaderValidation = DataLoader(ValidationDataset, batch_size=200, num_workers=0)

#%%
epoch = 200
modelTrained, regressionLogs = trainRegression(NNRegressorModel, featureLoaderTrain, featureLoaderValidation, epochs=epoch)

print(regressionLogs)

#%% 
from helperFunctions import plot_logs_regression

#%%
plot_logs_regression(regressionLogs)

#%% save model
import time
modelName="RegressionNNReg%d_%f.pth" % (epoch, time.time())
torch.save(modelTrained.state_dict(), modelPath+modelName)