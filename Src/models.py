#%%
import torch
from torch import nn
import torchvision.models as models
from copy import deepcopy
#%%
class NNRegressorDropout(nn.Module):
    def __init__(self,in_features, out_features=4,dropout=0,):
        super(NNRegressorDropout,self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features,436), #85 % di 512
            nn.RReLU(),
            nn.Dropout(0.2),
            nn.Linear(436,436), #apprendi un alto numero di feature
            nn.RReLU(),
            nn.Dropout(0.3),
            nn.Linear(436,380), #74% di 512
            nn.RReLU(),
            nn.Dropout(0.4),
            nn.Linear(380,260), #60% di 512
            nn.RReLU(),
            nn.Dropout(0.2),
            nn.Linear(260,102), #circa il 20% di 512 #regola 80-20
            nn.RReLU(),
            nn.Linear(102,4),      
        )
    def forward(self,x):
        return self.model(x)

class NNRegressor(nn.Module):
    def __init__(self,in_features, out_features=4):
        super(NNRegressor,self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features,400),
            nn.RReLU(),
            nn.Linear(400,300),
            nn.ReLU(),
            nn.Linear(300,200),
            nn.ReLU(),
            nn.Linear(200,100),
            nn.ReLU(),
            nn.Linear(100,100),
            nn.ReLU(),
            nn.Linear(100,50),
            nn.RReLU(),
            nn.Linear(50,out_features),
        )
    def forward(self,x):
        return self.model(x)

def getClassificationModel(out_features=16, previous_state_path=None):
    ResNet = models.resnet18(pretrained=True)
    model = deepcopy(ResNet)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs,out_features)
    if previous_state_path is not None:
        model.load_state_dict(torch.load(previous_state_path))
    model.double()
    return model