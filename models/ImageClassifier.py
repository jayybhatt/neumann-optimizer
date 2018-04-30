import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from modules.Net import Net
#from mlp import MLP
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from optimizer.neumann import Neumann

# Random seed
torch.manual_seed(1)
np.random.seed(1)

# Batch size
batch_size = 4

# Transformation to tensor and normalization
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

# Download the training set
trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)

# Training set loader
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=2)

# Test set
testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

# Test set loader
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)


# Function to show an image
def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
# end imshow


# Classes
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')

# Dataset as iterator
dataiter = iter(trainloader)

# Get next batch
images, labels = dataiter.next()

# Show images
n_batches = len(dataiter)
print(u"First 4 labels {}".format([classes[labels[j]] for j in range(4)]))
# imshow(torchvision.utils.make_grid(images))

# Our neural net
net = Net()

# uncomment below line if running on GPU
#net.cuda()

# Objective function is cross-entropy
criterion = nn.CrossEntropyLoss()

# Learning rate
learning_rate = 0.001

# Stochastic Gradient Descent
# optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
optimizer = Neumann(list(net.parameters()), lr=learning_rate, momentum = 0.9)

# Nb iterations
n_iterations = 30

# List of training and test accuracies
train_accuracies = np.zeros(n_iterations)
test_accuracies = np.zeros(n_iterations)

# Training !
for epoch in range(n_iterations):
    # Average loss during training
    average_loss = 0.0

    # Data to compute accuracy
    total = 0
    success = 0

    # Iterate over batches
    for i, data in enumerate(trainloader, 0):
        # Get the inputs and labels
        inputs, labels = data

        # To variable
        #inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        inputs, labels = Variable(inputs), Variable(labels)

        # Put grad to zero
        optimizer.zero_grad()

        # Forward
        outputs = net(inputs)

        loss = criterion(outputs, labels)

        # Backward
        loss.backward()

        # Optimize
        optimizer.step()

        # Add to loss
        average_loss += loss

        # Take the max as predicted
        _, predicted = torch.max(outputs.data, 1)

        # Add to total
        total += labels.size(0)

        # Add correctly classified images
        success += (predicted == labels.data).sum()
    # end for
    train_accuracy = 100.0 * success / total

    # Test model on test set
    success = 0
    total = 0
    for (inputs, labels) in testloader:
        # To variable
        #inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        inputs, labels = Variable(inputs), Variable(labels)
        # Neural net's output
        outputs = net(inputs)

        # Take the max is predicted
        _, predicted = torch.max(outputs.data, 1)

        # Add to total
        total += labels.size(0)

        # Add correctly classified images
        success += (predicted == labels.data).sum()
    # end for

    # Print average loss
    print(u"Epoch {}, average loss {}, train accuracy {}, test accuracy {}".format(
        epoch, average_loss / n_batches,
        train_accuracy,
        100.0 * success / total
        )
    )

    # Save the model
    train_accuracies[epoch] = train_accuracy
    test_accuracies[epoch] = 100.0 * success / total
# end for

plt.plot(np.arange(1, n_iterations+1), train_accuracies)
plt.plot(np.arange(1, n_iterations+1), test_accuracies)
plt.show()
