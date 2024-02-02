## We first import the pytorch nn module and optimizer
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report
import numpy as np

## Below you can see one way of defining the model class, where each individual 
## layer is defined as an instance variable.

class VGG16(nn.Module):
    def __init__(self, num_classes=3):
        super(VGG16, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer7 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer8 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer9 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer10 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer11 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer12 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer13 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(7*7*512, 4096),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU())
        self.fc2= nn.Sequential(
            nn.Linear(4096, num_classes))
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = self.layer11(out)
        out = self.layer12(out)
        out = self.layer13(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

class AlexNet1(nn.Module):
    def __init__(self, num_classes):
        super(AlexNet1, self).__init__()
        # input channel 3, output channel 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        # relu non-linearity
        self.relu1 = nn.ReLU()
        # max pooling
        self.max_pool2d1 = nn.MaxPool2d(kernel_size=3, stride=2)
        # input channel 64, output channel 192
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU()
        self.max_pool2d2 = nn.MaxPool2d(kernel_size=3, stride=2)
        # input channel 192, output channel 384
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        # input channel 384, output channel 256
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU()
        # input channel 256, output channel 256
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.relu5 = nn.ReLU()
        self.max_pool2d5 = nn.MaxPool2d(kernel_size=3, stride=2)
        # adaptive pooling
        self.adapt_pool = nn.AdaptiveAvgPool2d(output_size=(6, 6))
        #dropout layer
        self.dropout1 = nn.Dropout()
        # linear layer
        self.linear1 = nn.Linear(in_features=9216, out_features=4096, bias=True)
        self.relu6 = nn.ReLU()
        self.dropout2 = nn.Dropout()
        self.linear2 = nn.Linear(in_features=4096, out_features=4096, bias=True)
        self.relu7 = nn.ReLU()
        self.linear3 = nn.Linear(in_features=4096, out_features=num_classes, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.max_pool2d1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.max_pool2d2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.conv5(x)
        x = self.relu5(x)
        x = self.max_pool2d5(x)
        x = self.adapt_pool(x)
        # Note how we are flattening the feature map, B x C x H x W -> B x C*H*W
        x = x.reshape(x.shape[0], -1)
        x = self.dropout1(x)
        x = self.linear1(x)
        x = self.relu6(x)
        x = self.dropout2(x)
        x = self.linear2(x)
        x = self.relu7(x)
        x = self.linear3(x)
        return x

## Below you can see another way of defining the model class, where some layers  
## together are defined as an instance variable.
class AlexNet2(nn.Module):
    def __init__(self, num_classes):
        super(AlexNet2, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.AdaptiveAvgPool2d(output_size=(6, 6))
            )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(in_features=9216, out_features=4096, bias=True),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=num_classes, bias=True)
            )

    def forward(self, x):
        x = self.features(x)
        # Note how we are flattening the feature map, B x C x H x W -> B x C*H*W
        x = x.reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x

## Below we define the model as defined in the torchvision package and haven't 
## initialised with pretrained weights (see pretrained=False flag)
class AlexNet3(nn.Module):
    def __init__(self, num_classes):
        super(AlexNet3, self).__init__()
        from torchvision import models
        alexnet = models.alexnet(weights=None)
        self.features = alexnet.features
        self.avgpool = alexnet.avgpool
        self.classifier = alexnet.classifier
        # Please note how to change the last layer of the classifier for a new dataset
        # ImageNet-1K has 1000 classes, but STL-10 has 10 classes
        self.classifier[6] = nn.Linear(in_features=4096, out_features=num_classes, bias=True)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        # Note how we are flattening the feature map, B x C x H x W -> B x C*H*W
        x = x.reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x

## Below we define the model as defined in the torchvision package and initialised 
## with pretrained weights (see pretrained=True flag)
class AlexNet4(nn.Module):
    def __init__(self, num_classes):
        super(AlexNet4, self).__init__()
        from torchvision import models
        alexnet = models.alexnet(weights='IMAGENET1K_V1')
        self.features = alexnet.features
        self.avgpool = alexnet.avgpool
        self.classifier = alexnet.classifier
        # Please note how to change the last layer of the classifier for a new dataset
        # ImageNet-1K has 1000 classes, but STL-10 has 10 classes
        self.classifier[6] = nn.Linear(in_features=4096, out_features=num_classes, bias=True)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        # Note how we are flattening the feature map, B x C x H x W -> B x C*H*W
        x = x.reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


from tqdm.notebook import tqdm
##define train function
def train(model, device, train_loader, optimizer):
    # meter
    loss = AverageMeter()
    acc = AverageMeter()
    correct = 0
    # switch to train mode
    model.train()
    tk0 = tqdm(train_loader, total=int(len(train_loader)))
    for batch_idx, (data, target) in enumerate(tk0):
    # for data, target in train_loader:
        # print(batch_idx)
        # after fetching the data transfer the model to the 
        # required device, in this example the device is gpu
        # transfer to gpu can also be done by 
        # data, target = data.cuda(), target.cuda()
        data, target = data.to(device), target.to(device)  
        # compute the forward pass
        # it can also be achieved by model.forward(data)
        output = model(data) 
        # compute the loss function
        loss_this = F.mse_loss(output, target)
        # initialize the optimizer
        optimizer.zero_grad()
        # compute the backward pass
        loss_this.backward()
        # update the parameters
        optimizer.step()
        # update the loss meter 
        loss.update(loss_this.item(), target.shape[0])
        # get the index of the max log-probability
        pred = output.argmax(dim=1, keepdim=True) 
        # check which of the predictions are correct
        correct_this = pred.eq(target.view_as(pred)).sum().item()
        # accumulate the correct ones
        correct += correct_this
        # compute accuracy
        acc_this = correct_this/target.shape[0]*100.0
        # update the loss and accuracy meter 
        acc.update(acc_this, target.shape[0])


        # if batch_idx%10 == 0:
        # print("jk")
        # print('Train: Average loss: {:.4f}\n'.format(loss.avg)) 
        if batch_idx % 10 == 0:
            print('train: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(loss.avg, correct, len(train_loader.dataset), acc.avg))

    print('Train: Average loss: {:.4f}\n, Accuracy: {}/{} ({:.2f}%)\n'.format(loss.avg, correct, len(train_loader.dataset), acc.avg))

    return loss.avg

def arr_1(arr):
    r = []
    for i in arr:
        r.append(i[0])
    return r       

##define test function
def test(model, device, test_loader):
    # meters
    loss = AverageMeter()
    acc = AverageMeter()
    correct = 0
    # switch to test mode
    model.eval()
    ground_truth = []
    np_array = []
    for data, target in test_loader:
        # after fetching the data transfer the model to the 
        # required device, in this example the device is gpu
        # transfer to gpu can also be done by 
        # data, target = data.cuda(), target.cuda()
        data, target = data.to(device), target.to(device)  # data, target = data.cuda(), target.cuda()
        # since we dont need to backpropagate loss in testing,
        # we dont keep the gradient
        with torch.no_grad():
            # compute the forward pass
            # it can also be achieved by model.forward(data)
            output = model(data)
        # compute the loss function just for checking
        loss_this = F.mse_loss(output, target) # sum up batch loss
        # get the index of the max log-probability
        pred = output.argmax(dim=1, keepdim=True) 

        ground_truth.extend(np.argmax(output.to('cpu').numpy(),axis=-1))
        np_array.extend(arr_1(target.to('cpu').numpy().astype('int64')))

        # check which of the predictions are correct
        correct_this = pred.eq(target.view_as(pred)).sum().item()
        # accumulate the correct ones
        correct += correct_this
        # compute accuracy
        acc_this = correct_this/target.shape[0]*100.0
        # update the loss and accuracy meter 
        acc.update(acc_this, target.shape[0])
        loss.update(loss_this.item(), target.shape[0])

    print('Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(loss.avg, correct, len(test_loader.dataset), acc.avg))
    
    print("Classification report:")
    # y_pred = model.predict(X_test)

    print(classification_report(ground_truth, np_array, labels=[0,1,2], target_names=['Absent', 'Present', 'Unknown']))
    # print("Test accuracy:",accuracy_score(target, output))

    labels = [0,1,2]
    
    matrix = confusion_matrix(ground_truth, np_array)
    print(pd.DataFrame(matrix,columns=labels, index=labels))
    fig, ax = plt.subplots(figsize=(10, 10))

    # plot the confusion matrix with labels
    cm_display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels)

    # plot the confusion matrix on the specified axes object
    cm_display.plot(ax=ax)
    plt.show()

    

