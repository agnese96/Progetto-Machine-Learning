
#%%
import torch
from torch import nn
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.models as models
from copy import deepcopy

from loadData import ImageDataset

path = '..'
modelPath="C:/Users/beaut/Google Drive/Trio++/3°ANNO/Machine Learning/Progetto/Models/"

def get_vector(img):
    with torch.no_grad():
        t_img = Variable(img).unsqueeze(0).cuda()
    features = []
    features.append(model(t_img).data)
    return features

#%%
transform = transforms.Compose([transforms.Resize([224,224]),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
datasetTrain = ImageDataset(path+'/Dataset/images', path+'/Dataset/training_list.csv', transform=transform)
datasetValidation = ImageDataset(path+'/Dataset/images', path+'/Dataset/validation_list.csv', transform=transform)

#%%
ResNet = models.resnet18(pretrained=True)
model = deepcopy(ResNet) #copia modello 
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs,16)
model.load_state_dict(torch.load(modelPath+'ResNet18CrossEntropyReg10_1532170755.190167.pth'))
model.fc = nn.Linear(num_ftrs,512)
model.double()
model.cuda()

#%%
model.eval()
CNNOutputTrain = [get_vector(x['image']) for x in datasetTrain]
CNNOutputValidation = [get_vector(x['image']) for x in datasetValidation]

#%%
torch.save(CNNOutputTrain,modelPath+'FeaturesResNet18Train512.pth')
torch.save(CNNOutputValidation,modelPath+'FeaturesResNet18Validation512.pth')
