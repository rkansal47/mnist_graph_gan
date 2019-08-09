import setGPU
import torch
import torchvision
import torch.nn as nn
from torch.optim import Adam
from gcn import GCN_classifier
from graph_dataset_mnist import MNISTGraphDataset

batch_size = 128
num_thresholded = 100

transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

X_test = MNISTGraphDataset(num_thresholded, train=False)
# X_train = MNISTGraphDataset(num_thresholded, train=True)

# X_train_loaded = torch.utils.data.DataLoader(X_train, shuffle=True, batch_size=batch_size)
X_test_loaded = torch.utils.data.DataLoader(X_test, shuffle=False, batch_size=batch_size)

model = GCN_classifier(3, 256, 10, 0.3)

loss_func = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

for e in range(5):
    print(e)
    running_loss = 0
    for i, data in enumerate(X_test_loaded):
        print(i)
        input, labels = data
        optimizer.zero_grad()
        outputs = model(input)
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if(i%2000==0):
            print('[%d, %5d] loss: %.3f' % (e + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print("Done")
