from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


# Neural net
class Net(nn.Module):
    """
    Neural net
    """

    # Constructor
    def __init__(self):
        """
        Constructor
        """
        super(Net, self).__init__()
        self.conv_layer1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv_layer2 = nn.Conv2d(6, 16, 5)
        self.linear_layer1 = nn.Linear(16 * 4 * 4, 120)
        self.linear_layer2 = nn.Linear(120, 10)
    # end __init__

    # Forward pass
    def forward(self, x):
        """
        Forward pass
        :param x:
        :return:
        """
        # print(u"Input : {}".format(x.size()))
        x = self.conv_layer1(x)
        # print(u"Conv1 : {}".format(x.size()))
        x = F.relu(x)
        # print(u"Relu : {}".format(x.size()))
        x = self.pool(x)
        # print(u"Max pool : {}".format(x.size()))
        x = self.conv_layer2(x)
        # print(u"Conv2 : {}".format(x.size()))
        x = F.relu(x)
        x = self.pool(x)
        # print(u"Input : {}".format(x.size()))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.linear_layer1(x))
        x = F.relu(self.linear_layer2(x))
        return x
    # end forward

# end Net
