import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader,sampler,Dataset
import torchvision.datasets as dset
import torchvision.transforms as T
from PIL import Image
import os
import numpy as np
import scipy.io

import matplotlib.pyplot as plt
from torch.optim.optimizer import Optimizer
from torch.optim.sgd import SGD
from optimizer import Neumann

import _pickle as pkl

import pdb; pdb.set_trace()


label_mat=scipy.io.loadmat('../data/q3_2_data.mat')
label_train=label_mat['trLb']
print(len(label_train))
label_val=label_mat['valLb']
print(len(label_val))


class ActionDataset(Dataset):
    """Action dataset."""

    def __init__(self,  root_dir,labels=[], transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            labels(list): labels if images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.length=len(os.listdir(self.root_dir))
        self.labels=labels
    def __len__(self):
        return self.length*3

    def __getitem__(self, idx):

        folder=idx//3+1
        imidx= idx%3+1
        folder=format(folder,'05d')
        imgname=str(imidx)+'.jpg'
        img_path = os.path.join(self.root_dir,
                                folder,imgname)
        image = Image.open(img_path)
        if len(self.labels)!=0:
            Label=self.labels[idx//3][0]-1
        if self.transform:
            image = self.transform(image)
        if len(self.labels)!=0:
            sample={'image':image,'img_path':img_path,'Label':Label}
        else:
            sample={'image':image,'img_path':img_path}
        return sample



dtype = torch.FloatTensor # the CPU datatype
# Constant to control how frequently we print train loss
print_every = 400
# This is a little utility that we'll use to reset the model
# if we want to re-initialize all our parameters
def reset(m):
    if hasattr(m, 'reset_parameters'):
        m.reset_parameters()

class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size() # read in N, C, H, W
        return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image
gpu_dtype = torch.cuda.FloatTensor




def train(model, loss_fn, optimizer, dataloader, num_epochs = 1):
    losses = []
    for epoch in range(num_epochs):
        print('Starting epoch %d / %d' % (epoch + 1, num_epochs))
        model.train()
        for t, sample in enumerate(dataloader):
            x_var = Variable(sample['image'].cuda())
            y_var = Variable(sample['Label'].cuda().long())

            scores = model(x_var)

            loss = loss_fn(scores, y_var)
            if (t + 1) % 1 == 0:
                print('t = %d, loss = %.4f' % (t + 1, loss.data[0]))
                pass

            losses.append(loss.data[0])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return losses

def check_accuracy(model, loader):
    '''
    if loader.dataset.train:
        print('Checking accuracy on validation set')
    else:
        print('Checking accuracy on test set')
    '''
    num_correct = 0
    num_samples = 0
    model.eval() # Put the model in test mode (the opposite of model.train(), essentially)
    for t, sample in enumerate(loader):
        x_var = Variable(sample['image'].cuda())
        y_var = sample['Label'].cuda()
        y_var=y_var.cpu()
        scores = model(x_var)
        _, preds = scores.data.cpu().max(1)
        #print(preds)
        #print(y_var)
        num_correct += (preds.numpy() == y_var.numpy()).sum()
        num_samples += preds.size(0)
    acc = float(num_correct) / num_samples
    print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))




augment_transforms = T.Compose([T.RandomHorizontalFlip(),T.RandomVerticalFlip(),T.RandomRotation(30),T.ToTensor()])
batch_size = 256
print_every = 50
image_dataset_train=ActionDataset(root_dir='../data/trainClips/',labels=label_train,transform=augment_transforms)

image_dataloader_train = DataLoader(image_dataset_train, batch_size=batch_size,
                        shuffle=True, num_workers=4)
image_dataset_val=ActionDataset(root_dir='../data/valClips/',labels=label_val,transform=augment_transforms)

image_dataloader_val = DataLoader(image_dataset_val, batch_size=batch_size,
                        shuffle=False, num_workers=4)
image_dataset_test=ActionDataset(root_dir='../data/testClips/',labels=[],transform=augment_transforms)

image_dataloader_test = DataLoader(image_dataset_test, batch_size=batch_size,
                        shuffle=False, num_workers=4)



###########3rd To Do (16 points, must submit the results to Kaggle) ##############
# Train your model here, and make sure the output of this cell is the accuracy of your best model on the
# train, val, and test sets. Here's some code to get you started. The output of this cell should be the training
# and validation accuracy on your best model (measured by validation accuracy).

model = nn.Sequential(
            nn.Conv2d(3,32,kernel_size=5,stride=1), #8*58*58
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),#8*29*29

            nn.Conv2d(32,128,kernel_size=3,stride=2),#16*23*23, 15
            nn.BatchNorm2d(128),
            # nn.LeakyReLU(inplace=True),
            # nn.Dropout2d(p=0.4),
            # nn.MaxPool2d(kernel_size=2,stride=2),#16*11*11

            # nn.Conv2d(128,256,kernel_size=3,stride=1),
            # nn.BatchNorm2d(256),
            # nn.LeakyReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2,stride=2),
            Flatten(),
            nn.Linear(25088,10)
)

model.cuda()
model.apply(reset)
loss_fn = nn.CrossEntropyLoss().cuda().type(gpu_dtype)
# loss_fn = nn.CrossEntropyLoss()
beta = 1e-9
alpha = 1e-3
optimizer = Neumann(list(model.parameters()), lr=1e-3, alpha=alpha, beta=beta, sgd_steps=10)



num_epochs=5
# for i in range(1):
model.train()
losses = train(model, loss_fn, optimizer,image_dataloader_train, num_epochs=num_epochs)

model.eval()
check_accuracy(model,image_dataloader_train)
check_accuracy(model, image_dataloader_val)

filename = "./"+str(batch_size)+"_"+str(num_epochs)+".pkl"

with open(filename, 'wb') as f:
    pkl.dump(losses, f)

plt.figure(figsize=(12, 8))
plt.title("Neumann Opt on Action Detection", fontsize=17)
plt.xlabel("Iteration", fontsize=15)
plt.ylabel("Loss", fontsize=15)
plt.plot(np.arange(len(losses)), losses)
plt.show()


