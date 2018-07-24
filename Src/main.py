#%% usare se si hanno problemi di path per entrare dentro la cartella Src
import os
#os.chdir('./Src')
path = os.getcwd()
path

#%% serve per jupyter
import os
os.chdir('./Src')
path='..'

#%% main imports
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np

#%%
from loadData import ImageDataset

#%%
transform = transforms.Compose([transforms.Resize([224,224]),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
datasetTrain = ImageDataset(path+'/Dataset/images', path+'/Dataset/training_list.csv', transform=transform)

#%%
datasetValidation = ImageDataset(path+'/Dataset/images', path+'/Dataset/validation_list.csv', transform=transform)

#%% definiamo i data loaders
imageLoaderTrain = DataLoader(datasetTrain, batch_size=5, num_workers=0, shuffle=True)
imageLoaderValidation = DataLoader(datasetValidation, batch_size=5, num_workers=0)

#%%
import torchvision.models as models
ResNet = models.resnet18(pretrained=True)

#%%
#print(ResNet) #visualizza modello 

#%%
from copy import deepcopy
model = deepcopy(ResNet) #copia modello 

#%%
from torch import nn
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(nn.Linear(num_ftrs,100),nn.Dropout(),nn.ReLU(),nn.Linear(100,4))

#%%
model.double()

#%%
torch.cuda.empty_cache()

#%%
modelPath="C:/Users/beaut/Google Drive/Trio++/3°ANNO/Machine Learning/Progetto/Models/"
#model.load_state_dict(torch.load(modelPath+'ResNet18LocalizationLossReg5_1532099229.886488.pth'))

#%%
from trainFunction import trainRegression
epoch = 1
#%%
modelTrained, regressionLogs = trainRegression(model, imageLoaderTrain, imageLoaderValidation, epochs=epoch)

print(regressionLogs)

#%% save model
import time
modelName="ResNet18LocalizationLossDropout%d_%f.pth" % (epoch, time.time())
torch.save(modelTrained.state_dict(), modelPath+modelName)
#torch.save(regressionLogs, modelPath+'/Logs/'+modelName)

#%% 
#from helperFunctions import plot_logs_classification

#%%
#plot_logs_classification(regressionLogs)