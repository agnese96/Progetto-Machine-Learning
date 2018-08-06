#%%
import os
os.chdir('./Src')
path='..'

#%%
modelPath="C:/Users/beaut/Google Drive/Trio++/3°ANNO/Machine Learning/Progetto/Models/"

#%%
#modelPath="/Users/alessandrodistefano/GoogleDrive/Trio++/3°ANNO/Machine\ Learning/Progetto/Models"
#print (modelPath)

#%%
import torch 
from loadData import FeatureDataset, ImageDataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

#%% 
CNNOutputTrain = torch.Tensor(torch.load(modelPath+'FeaturesResNet18Train512.pth')).unsqueeze(1)
CNNOutputValidation = torch.Tensor(torch.load(modelPath+'FeaturesResNet18Validation512.pth')).unsqueeze(1)
#%%
TrainDataset = FeatureDataset(CNNOutputTrain, path+'/Dataset/training_list.csv')
ValidationDataset = FeatureDataset(CNNOutputValidation, path+'/Dataset/validation_list.csv')

#%%
from models import NNRegressor, NNRegressorDropout
NNRegressorModel = NNRegressorDropout(512,4)
NNRegressorModel.double()

#%% definiamo i data loaders
featureLoaderTrain = DataLoader(TrainDataset, batch_size=200, num_workers=0, shuffle=True)
featureLoaderValidation = DataLoader(ValidationDataset, batch_size=200, num_workers=0)

#%%
NNRegressorModel.load_state_dict(torch.load(modelPath+'ale_models/Progetto-Machine-LearningFinalPostReg1Ale_RegressionNNReg50_1532643401.950671.pth'))

#%%
from trainFunction import trainRegression
epochs = 200
lr = 0.00021
modelTrained, regressionLogs = trainRegression(NNRegressorModel, featureLoaderTrain, featureLoaderValidation, epochs=epochs, lr=lr,)
print(regressionLogs)

#%% save model
import time
modelName="RegressionNNDropoutAleBeta0.85%d_%f.pth" % (epochs, time.time())
torch.save(modelTrained.state_dict(), modelPath+modelName)

#%% 
from helperFunctions import plot_logs_regression
plot_logs_regression(regressionLogs)
#%%
# from models import NNRegressorDropout
# modelTrained = NNRegressorDropout(512,4)
# modelTrained.double()
# modelTrained.load_state_dict(torch.load(modelPath+'ale_models/Progetto-Machine-LearningFinalPostReg3Ale_RegressionNNReg5_1532649162.319779.pth'))
#%%
from helperFunctions import predict, get_gt
import numpy as np
predictions = predict(modelTrained,ValidationDataset,'features')
gt = get_gt(ValidationDataset,'target')

#%%
from evaluate import evaluate_localization
errors = evaluate_localization(predictions,gt)
print("Errors:")
print("Mean Location Error: %0.4f" % (errors[0],))
print("Median Location Error: %0.4f" % (errors[1],))
print("Mean Orientation Error: %0.4f" % (errors[2],))
print("Median Orientation Error: %0.4f" % (errors[3],))
