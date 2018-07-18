#%% usare se si hanno problemi di path per entrare dentro la cartella Src
import os
os.chdir('./Src')

#%% main imports
import torch
from loadData import ImageDataset
import torchvision.transforms as transforms
#%%
datasetTrain = ImageDataset('../Dataset/images', '../Dataset/training_list.csv', transform=transforms.ToTensor())
sample = datasetTrain[0]
print(sample['image'].shape)
print(sample['features'])