#%%
import os
os.chdir('./Src')
path='..'

#%%
modelPath="C:/Users/enric/Google Drive/Trio++/3°ANNO/Machine Learning/Progetto/Models/"

#%%
#modelPath="/Users/alessandrodistefano/GoogleDrive/Trio++/3°ANNO/Machine\ Learning/Progetto/Models"
#print (modelPath)

#%%
import torch 
from loadData import FeatureDataset, ImageDataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

#%% 
CNNOutputTrain = torch.load(modelPath+'FeaturesResNet18Train512.pth')
CNNOutputValidation = torch.load(modelPath+'FeaturesResNet18Validation512.pth')

#%%
TrainDataset = FeatureDataset(CNNOutputTrain, path+'/Dataset/training_list.csv')
ValidationDataset = FeatureDataset(CNNOutputValidation, path+'/Dataset/validation_list.csv')

#%%
from models import NNRegressor, NNRegressorDropout
NNRegressorModel = NNRegressor(512,4)
NNRegressorModel.double()

#%% definiamo i data loaders
featureLoaderTrain = DataLoader(TrainDataset, batch_size=200, num_workers=0, shuffle=True)
featureLoaderValidation = DataLoader(ValidationDataset, batch_size=200, num_workers=0)

#%%
#NNRegressorModel.load_state_dict(torch.load(modelPath+'RegressionNNLowerLRMomentumDropout200_1532540427.751633.pth'))

#%%
from trainFunction import trainRegression
epochs = 400
modelTrained, regressionLogs = trainRegression(NNRegressorModel, featureLoaderTrain, featureLoaderValidation, epochs=epochs)
print(regressionLogs)

#%% save model
import time
modelName="RegressionNNLowerLRMomentumRReLuReg%d_%f.pth" % (epochs, time.time())
torch.save(modelTrained.state_dict(), modelPath+modelName)

#%% 
from helperFunctions import plot_logs_regression
plot_logs_regression(regressionLogs)

#%%
from helperFunctions import predict
import numpy as np
predictions = predict(modelTrained,ValidationDataset,'features')
gt = []
for x in ValidationDataset:
    gt.append(x['target'])
gt = np.array(gt)

#%%
from evaluate import evaluate_localization
errors = evaluate_localization(predictions,gt)
print("Errors:")
print("Mean Location Error: %0.4f" % (errors[0],))
print("Median Location Error: %0.4f" % (errors[1],))
print("Mean Orientation Error: %0.4f" % (errors[2],))
print("Median Orientation Error: %0.4f" % (errors[3],))
