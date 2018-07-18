#%% usare se si hanno problemi di path per entrare dentro la cartella Src
import os
#os.chdir('./Src')
path = os.getcwd()

#%% serve per jupyter
#import os
#os.chdir('./Src')
#path='..'

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
imageLoaderTrain = DataLoader(datasetTrain, batch_size=10, num_workers=0, shuffle=True)
imageLoaderValidation = DataLoader(datasetValidation, batch_size=10, num_workers=2)

#%%
import torchvision.models as models
squeezenet = models.squeezenet1_1(pretrained=True)
#%%
print(squeezenet) #visualizza modello 

#%%
from copy import deepcopy
model = deepcopy(squeezenet) #copia modello 

#%%
from torch import nn
classifierMod = list(model.classifier)
classifierMod.append(nn.Linear(1000,16))
#%%
model.classifier = nn.Sequential(*classifierMod)
#%%
""" from torch import nn
fcMod = [model.fc, nn.Linear(1000,16)]
model.fc = nn.Sequential(*fcMod)

#%% 
model.fc """

#%%
torch.cuda.empty_cache()
#%%
from trainFunction import trainClassification
modelTrained, classificationLogs = trainClassification(model, imageLoaderTrain, imageLoaderValidation)

print(modelTrained.shape)
