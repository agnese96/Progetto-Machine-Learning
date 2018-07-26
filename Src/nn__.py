#%%
import os
os.chdir('./Src')
path='..'
#%%
modelPath="C:/Users/beaut/Google Drive/Trio++/3°ANNO/Machine Learning/Progetto/Models/"
#%%
modelPath="/Users/alessandrodistefano/GoogleDrive/Trio++/3°ANNO/Machine\ Learning/Progetto/Models"
print (modelPath)
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

from models import NNRegressorDropout
NNRegressorModel = NNRegressorDropout(512,4)
NNRegressorModel.double()
#%% definiamo i data loaders
featureLoaderTrain = DataLoader(TrainDataset, batch_size=200, num_workers=0, shuffle=True)
featureLoaderValidation = DataLoader(ValidationDataset, batch_size=200, num_workers=0)

#%%
from trainFunction import trainRegression
epoch = 200
modelTrained, regressionLogs = trainRegression(NNRegressorModel, featureLoaderTrain, featureLoaderValidation, epochs=epoch)
print(regressionLogs)

#%% 
from helperFunctions import plot_logs_regression
plot_logs_regression(regressionLogs)

#%%
modelTrained = NNRegressorDropout(512,4)
modelTrained.double()
modelTrained.load_state_dict(torch.load(modelPath+'RegressionNNLowerLRMomentumDropout200_1532540427.751633.pth'))
#%%
from helperFunctions import predict
predictions = predict(modelTrained,ValidationDataset,'features')
gt = []
for x in ValidationDataset:
    gt.append(x['target'].data)
#%%
print(predictions[0],gt[0])
#%%
from evaluate import evaluate_localization
position_error = evaluate_localization(predictions,gt)
#%% save model
import time
modelName="RegressionNNLowerLRMomentumDropout%d_%f.pth" % (epoch, time.time())
torch.save(modelTrained.state_dict(), modelPath+modelName)