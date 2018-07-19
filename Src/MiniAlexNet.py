from torch import nn
class MiniAlexNet(nn.Module):
  def __init__(self, input_channels=3, out_classes=100):
    super(MiniAlexNet, self).__init__() 
    #ridefiniamo il modello utilizzando i moduli sequential.
    #ne definiamo due: un "feature extractor", che estrae le feature maps
    #e un "classificatore" che implementa i livelly FC
    self.feature_extractor = nn.Sequential(
      #Conv1
      nn.Conv2d(input_channels, 16, 5, padding=2), #Input: 3 x 32 x 32. Ouput: 16 x 32 x 32
      nn.MaxPool2d(2), #Input: 16 x 32 x 32. Output: 16 x 16 x 16
      nn.ReLU(),
                  
      #Conv2
      nn.Conv2d(16, 32, 5, padding=2), #Input 16 x 16 x 16. Output: 32 x 16 x 16
      nn.MaxPool2d(2), #Input: 32 x 16 x 16. Output: 32 x 8 x 8
      nn.ReLU(),
                  
      #Conv3
      nn.Conv2d(32, 64, 3, padding=1), #Input 32 x 8 x 8. Output: 64 x 8 x 8
      nn.ReLU(),
                  
      #Conv4
      nn.Conv2d(64, 128, 3, padding=1), #Input 64 x 8 x 8. Output: 128 x 8 x 8
      nn.ReLU(),
                  
      #Conv5
      nn.Conv2d(128, 256, 3, padding=1), #Input 128 x 8 x 8. Output: 256 x 8 x 8
      nn.MaxPool2d(2), #Input: 256 x 8 x 8. Output: 256 x 4 x 4
      nn.ReLU()
    )
            
    self.classifier = nn.Sequential(
      #FC6
      nn.Linear(4096, 2048), #Input: 256 * 4 * 4
      nn.ReLU(),
                  
      #FC7
      nn.Linear(2048, 1024),
      nn.ReLU(),
                  
      #FC8
      nn.Linear(1024, out_classes)
    )
          
  def forward(self,x):
    #Applichiamo le diverse trasformazioni in cascata
    x = self.feature_extractor(x)
    x = self.classifier(x.view(x.shape[0],-1))
    return x