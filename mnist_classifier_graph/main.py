import torch
import torch.nn.functional as F
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, num_thresholded, dropout, batch_size):
        super(Classifier, self).__init__()



        self.max_pool = nn.MaxPool2d(2)
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, input):
        x = self.reformat(input)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = F.sigmoid(x)
        return x

    def reformat(self, input):
        out = torch.zeros((28, 28))
