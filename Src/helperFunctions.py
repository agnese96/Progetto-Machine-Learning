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
    model.eval()
    predictions = []
    for x in dataset:
        with torch.no_grad():
            temp = Variable(x[input_key]).unsqueeze(0).cuda()
        prediction = np.array(model(temp).squeeze().data)
        predictions.append(prediction)
    predictions = np.array(predictions)
    return predictions

def predictLabel(model, dataset, input_key='image'):
    predictions = predict(model,dataset,input_key)
    label = np.argmax(predictions, axis=1)
    print(label)
    return label


def get_gt(dataset,input_key):
    gt = []
    for x in dataset:
        gt.append(x[input_key])
    return np.array(gt)
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

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

