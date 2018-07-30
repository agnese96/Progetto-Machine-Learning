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
transform = transforms.Compose([transforms.Resize([224,224]),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])])
datasetTrain = ImageDataset(path+'/Dataset/images', path+'/Dataset/training_list.csv', transform=transform)

#%%
datasetValidation = ImageDataset(path+'/Dataset/images', path+'/Dataset/validation_list.csv', transform=transform)

#%% definiamo i data loaders
imageLoaderTrain = DataLoader(datasetTrain, batch_size=5, num_workers=0, shuffle=True)
imageLoaderValidation = DataLoader(datasetValidation, batch_size=5, num_workers=0)

#%%
from models import getClassificationModel
modelPath="C:/Users/beaut/Google Drive/Trio++/3°ANNO/Machine Learning/Progetto/Models/old models/"
modelName = modelPath+'ResNet18CrossEntropyReg10_1532170755.190167.pth'
model = getClassificationModel(previous_state_path=modelName)

#%%
torch.cuda.empty_cache()
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

#%% 
from helperFunctions import plot_logs_classification, predictLabel, get_gt, plot_confusion_matrix
from sklearn.metrics import confusion_matrix,f1_score
#plot_logs_classification(regressionLogs)
predicted = predictLabel(model,datasetValidation)
gt = get_gt(datasetValidation,'label')
cm = confusion_matrix(gt,predicted)
#Vediamo come percentuali 
cm = cm.astype(float)/cm.sum(1).reshape(-1,1)
print(cm)
#%%
from helperFunctions import plot_confusion_matrix
plot_confusion_matrix(cm, list(range(16)))
#%%
scores = f1_score(gt,predicted,average=None)
print("F1 per ogni classe: " , scores)
print("F1 medio per calcolare la performance del classificatore: ", scores.mean())