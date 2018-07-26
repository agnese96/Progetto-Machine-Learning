#%%
import torch
from torch import nn
import torchvision.models as models
from copy import deepcopy
#%%
class NNRegressorDropout(nn.Module):
    def __init__(self,in_features, out_features=4,dropout=0.25,):
        super(NNRegressorDropout,self).__init__()
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
            nn.Dropout(p=dropout),
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