#%%
import os
os.chdir('./Src')
path='..'
#%%
modelPath="C:/Users/beaut/Google Drive/Trio++/3°ANNO/Machine Learning/Progetto/Models/"

#%% main imports
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np

#%%
from loadData import ImageDataset, FeatureDataset

#%%
transform = transforms.Compose([transforms.Resize([224,224]),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])])
                                                     
classificationDataset = ImageDataset(path+'/Dataset/images', path+'/Dataset/testing_list_blind.csv', transform=transform, mode='test')
imageFeatures = torch.Tensor(torch.load(modelPath+'FeaturesResNet18Test512.pth')).unsqueeze(1)
#%%
imageFeatures.shape
#%%
regressionDataset = FeatureDataset(imageFeatures, None)

#%%
from models import getClassificationModel, NNRegressorDropout
modelName = modelPath+'old models/ResNet18CrossEntropyReg10_1532170755.190167.pth'
classificationModel = getClassificationModel(previous_state_path=modelName)
regressionModel = NNRegressorDropout(512)

#%%
from helperFunctions import predict, predictLabel
predRegression = predict(regressionModel, regressionDataset, input_key='features')
predClassification = predictLabel(classificationModel, classificationDataset)
#%%
final_matrix = []
numElements = len(classificationDataset)
for i in range(numElements):
    final_matrix.append([classificationDataset[i]['path'],
                         *predRegression[i],
                         predClassification[i]])
final_matrix = np.stack(final_matrix).astype(str)
#%%
np.savetxt(modelPath+'outputTest.csv', final_matrix, delimiter=',', fmt="%s")