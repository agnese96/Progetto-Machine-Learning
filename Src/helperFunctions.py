#%%
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.autograd import Variable

#%% plottiamo i log
def plot_logs_classification(logs):
    train_losses, train_acc, test_losses, test_acc = \
        logs[0]['train'], logs[1]['train'], logs[0]['validation'], logs[1]['validation']
    plt.figure(figsize=(18,6))
    plt.subplot(121)
    plt.plot(train_losses)
    plt.plot(test_losses)
    plt.legend(['Training loss', 'Testing loss'])
    plt.grid()
    plt.subplot(122)
    plt.plot(train_acc)
    plt.plot(test_acc)
    plt.legend(['Training accuracy', 'Testing accuracy'])
    plt.grid()
    plt.show()

#%%
def predict(model, dataset, input_key):
    model.cuda()
    predictions = []
    for x in dataset:
        with torch.no_grad():
            temp = Variable(x[input_key]).unsqueeze(0).cuda()
        # do this to return list and not vector
        prediction = []
        prediction.append(model(temp).data)
        predictions.append(prediction)
    return predictions

#%% plottiamo i log
def plot_logs_regression(logs):
    train_losses, train_acc, test_losses, test_acc = \
        logs[0]['train'], logs[1]['train'], logs[0]['validation'], logs[1]['validation']
    plt.figure(figsize=(18,6))
    plt.subplot(121)
    plt.plot(train_losses)
    plt.plot(test_losses)
    plt.yscale('log')
    plt.legend(['Training loss', 'Testing loss'])
    plt.grid()
    plt.subplot(122)
    plt.plot(train_acc)
    plt.plot(test_acc)
    plt.yscale('log')
    plt.legend(['Training accuracy', 'Testing accuracy'])
    plt.grid()
    plt.show()

def rec_curve(predictions, gts):
    assert len(predictions) == len(gts)
    errors = []
    for pred, gt in zip(predictions, gts):
        errors.append((pred-gt)**2)
    bins = np.sort(np.unique(errors))
    freq = []
    for bin in bins:
        freq.append((errors <= bin).mean)
    AUC = np.trapz(freq,bins)
    totArea = np.max(bins)*1
    AOC = totArea - AUC
    _plot_rec_curve(bins, freq, AOC)

def _plot_rec_curve(bins,freq,AOC, name='REC curve'):
    plt.plot(bins, freq)
    plt.legend(['%s. AOC: %0.2f'%(name, AOC)])
    plt.grid()
    plt.show()


