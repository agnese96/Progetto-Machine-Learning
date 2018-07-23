#%%
import torch
from torch import nn 
from sklearn.metrics import mean_absolute_error
from torch.optim import SGD
from torch.autograd import Variable
from torch.utils.data import DataLoader 
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
    return ((x_pred-x)**2+(y_pred-y)**2 + beta*((u_pred-u)**2+(v_pred-v)**2)).sum()
def MSE(gt, predictions):
    return((predictions-gt)**2).mean()
#%%
def trainClassification(model, train_loader, test_loader, lr=0.01, epochs=20, momentum=0.9, weight_decay = 0.000001):
    criterion = localizationLoss
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
                target=Variable(batch["target"],)
                if torch.cuda.is_available(): 
                    input, target = input.cuda(), target.cuda()
                output = model(input)
                #output.data = output.data.float()
                l = criterion(output, target) 
                if mode=='train':
                    l.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                # print(output.data)
                # print(type(output.data))
                acc = MSE(target.data, output.data)
                #print(l.item(),input.shape[0])
                epoch_loss+=l.item()*input.shape[0]
                epoch_acc+=acc*input.shape[0]
                samples+=input.shape[0]      
            """ print("\r[%s] Epoch %d/%d. Iteration %d/%d. Loss: %0.2f. Accuracy: %0.2f\t\t\t\t\t" % \
            (mode, e+1, epochs, i, len(loaders[mode]), epoch_loss/samples, epoch_acc/samples),) """
            epoch_loss/=len(loaders[mode].dataset)
            epoch_acc/=len(loaders[mode].dataset)
            losses[mode].append(epoch_loss)
            accuracies[mode].append(epoch_acc)
            print ("\r[%s] Epoch %d/%d. Iteration %d/%d. Loss: %0.2f. Accuracy: %0.2f\t\t\t\t\t" % \
            (mode, e+1, epochs, i, len(loaders[mode]), epoch_loss, epoch_acc),)
    #restituiamo il modello e i vari log
    return model, (losses, accuracies)                                                                                                                                            