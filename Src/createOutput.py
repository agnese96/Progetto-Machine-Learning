#%%
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
transform = transforms.Compose([transforms.Resize([224,224]),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])])
                                                     
classificationDataset = ImageDataset(path+'/Dataset/images', path+'/Dataset/testing_list_blind.csv', transform=transform)

#%%
from models import getClassificationModel
modelPath="C:/Users/beaut/Google Drive/Trio++/3°ANNO/Machine Learning/Progetto/Models/old models/"
modelName = modelPath+'ResNet18CrossEntropyReg10_1532170755.190167.pth'
classificationModel = getClassificationModel(previous_state_path=modelName)

#%% TODO: add load test features regression after creating them with extractFeatures
 