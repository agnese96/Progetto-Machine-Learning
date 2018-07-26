#%%
import torch
from torch import nn 
from sklearn.metrics import accuracy_score, mean_absolute_error
from torch.optim import SGD
from torch.autograd import Variable
from torch.utils.data import DataLoader 

#%%
def trainClassification(model, train_loader, test_loader, lr=0.01, epochs=20, momentum=0.9, weight_decay = 0.000001):
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
                input=Variable(batch["image"], requires_grad=True)
                label=Variable(batch["label"],)
                if torch.cuda.is_available(): 
                    input, label = input.cuda(), label.cuda(async=True)
                output = model(input)
                l = criterion(output, label) 
                if mode=='train':
                    l.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                acc = accuracy_score(label.data, output.max(1)[1].data)
                epoch_loss+=l.item()*input.shape[0]
                epoch_acc+=acc*input.shape[0]
                samples+=input.shape[0]
                #print('Iteration: %d'%i)
            epoch_loss/=len(loaders[mode].dataset)
            epoch_acc/=len(loaders[mode].dataset)
            losses[mode].append(epoch_loss)
            accuracies[mode].append(epoch_acc)
            print ("\r[%s] Epoch %d/%d. Iteration %d/%d. Loss: %0.2f. Accuracy: %0.2f\t\t\t\t\t" % \
            (mode, e+1, epochs, i, len(loaders[mode]), epoch_loss, epoch_acc),)
    #restituiamo il modello e i vari log
    return model, (losses, accuracies) 
                                                                                                                                          
#%%
def localizationLoss(input, target, beta=0.7):
    x_pred = input[:,0]
    y_pred = input[:,1]
    u_pred = input[:,2]
    v_pred = input[:,3]
    x = target[:,0]
    y = target[:,1]
    u = target[:,2]
    v = target[:,3]
    return torch.mean((1-beta)*((x_pred-x)**2+(y_pred-y)**2) + beta*((u_pred-u)**2+(v_pred-v)**2))

def trainRegression(model, train_loader, test_loader, lr=0.001, epochs=20, momentum=0.9, weight_decay = 0.000001):
    criterion = localizationLoss
    optimizer = SGD(model.parameters(),lr, momentum=momentum)
    loaders = {'train':train_loader, 'validation':test_loader} 
    losses = {'train':[], 'validation':[]}
    mae_cumulative = {'train':[], 'validation':[]}
    if torch.cuda.is_available(): 
        model=model.cuda()
    for e in range(epochs):
        for mode in ['train', 'validation']:
            if mode=='train': 
                model.train()
            else: 
                model.eval()
            epoch_loss = 0
            epoch_mae = 0
            samples = 0
            for i, batch in enumerate(loaders[mode]):
                input=Variable(batch["features"], requires_grad=True)
                target=Variable(batch["target"],)
                if torch.cuda.is_available(): 
                    input, target = input.cuda(), target.cuda(async=True)
                output = model(input).squeeze(1)
                l = criterion(output, target)
                if mode=='train':
                    l.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                mae = mean_absolute_error(target.data, output.data)
                epoch_loss+=l.item()*input.shape[0]
                epoch_mae+=mae*input.shape[0]
                samples+=input.shape[0]
                #print('Iteration: %d'%i)
            epoch_loss/=len(loaders[mode].dataset)
            epoch_mae/=len(loaders[mode].dataset)
            losses[mode].append(epoch_loss)
            mae_cumulative[mode].append(epoch_mae)
            print ("\r[%s] Epoch %d/%d. Iteration %d/%d. Loss: %0.2f. Mean Absolute Error: %0.2f\t\t\t\t\t" % \
            (mode, e+1, epochs, i, len(loaders[mode]), epoch_loss, epoch_mae),)
    #restituiamo il modello e i vari log
    return model, (losses, mae_cumulative) 
