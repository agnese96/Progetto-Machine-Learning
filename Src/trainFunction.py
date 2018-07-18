#%%
import torch
from torch import nn 
from sklearn.metrics import accuracy_score
from torch.optim import SGD
from torch.autograd import Variable
from torch.utils.data import DataLoader 
#%%
def trainClassification(model,train_loader, test_loader, lr=0.01, epochs=10, momentum=0.9, weight_decay = 0.000001):
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(),lr, momentum=momentum, weight_decay=weight_decay)
    loaders = {'train':train_loader, 'validation':test_loader} 
    losses = {'train':[], 'validation':[]}
    accuracies = {'train':[], 'validation':[]}
    if torch.cuda.is_available(): 
        model=model.cuda()
    for e in range(epochs):
        for mode in ['train', 'validation']:
            if mode=='train': 
                model.train()
            else: 
                model.eval()
            epoch_loss = 0
            epoch_acc = 0
            samples = 0
            for i, batch in enumerate(loaders[mode]):
                x=Variable(batch["image"], requires_grad=(mode=='train'))
                y=Variable(batch["label"],)
                if torch.cuda.is_available(): 
                    x, y = x.cuda(), y.cuda()
                output = model(x)
                l = criterion(output,y) 
                if mode=='train':
                    l.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                acc = accuracy_score(y.data,output.max(1)[1].data)
                epoch_loss+=l.data[0]*x.shape[0]
                epoch_acc+=acc*x.shape[0]
                samples+=x.shape[0]      
            print("\r[%s] Epoch %d/%d. Iteration %d/%d. Loss: %0.2f. Accuracy: %0.2f\t\t\t\t\t" % \
            (mode, e+1, epochs, i, len(loaders[mode]), epoch_loss/samples, epoch_acc/samples),)
    epoch_loss/=samples
    epoch_acc/=samples
    losses[mode].append(epoch_loss)
    accuracies[mode].append(epoch_acc)
    print ("\r[%s] Epoch %d/%d. Iteration %d/%d. Loss: %0.2f. Accuracy: %0.2f\t\t\t\t\t" % \
    (mode, e+1, epochs, i, len(loaders[mode]), epoch_loss, epoch_acc),)
    #restituiamo il modello e i vari log
    return model, (losses, accuracies)                                                                                                                                            