#%% usare se si hanno problemi di path per entrare dentro la cartella Src
import os
#os.chdir('./Src')
path = os.getcwd()
path

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
imageLoaderTrain = DataLoader(datasetTrain, batch_size=5, num_workers=0, shuffle=True)
imageLoaderValidation = DataLoader(datasetValidation, batch_size=5, num_workers=0)

#%%
import torchvision.models as models
ResNet = models.resnet18(pretrained=True)

#%%
print(ResNet) #visualizza modello 

#%%
from copy import deepcopy
model = deepcopy(ResNet) #copia modello 

#%%
""" from torch import nn
classifierMod = list(model.classifier)
classifierMod.append(nn.ReLU(inplace=True))
classifierMod.append(nn.Linear(100,100))
classifierMod.append(nn.ReLU(inplace=True))
classifierMod.append(nn.Linear(1000, 16)) """
"""from torch import nn
classifierMod = list(model.fc)
classifierMod.pop()
classifierMod.append(nn.Linear(4096,16))"""

#%%
""""model.fc = nn.Sequential(*classifierMod)

#%%
model.fc"""

#%%
torch.cuda.empty_cache()

#%%
modelPath="C:/Users/enric/Google Drive/Trio++/3°ANNO/Machine Learning/Progetto/Models/"
#model.load_state_dict(torch.load(modelPath+'ResNet18CrossEntropyReg5_1532030250.256796.pth'))

#%%
from trainFunction import trainClassification
epoch = 5
modelTrained, classificationLogs = trainClassification(model, imageLoaderTrain, imageLoaderValidation, epochs=epoch)

print(classificationLogs)

#%% save model
import time
modelName="ResNet18CrossEntropyReg%d_%f.pth" % (epoch, time.time())
torch.save(modelTrained.state_dict(), modelPath+modelName)

#%% 
from helperFunctions import plot_logs_classification

#%%
plot_logs_classification(classificationLogs)