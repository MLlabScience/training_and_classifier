import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import data_loader
from model import Net

# Load data
trainloader, testloader, classes = data_loader.get_data_loaders(batch_size=4)

# Define the network
net = Net()

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Training loop
for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # ...

        # Training code here (as in your existing code)

        # ...

        if i % 2000 == 1999:
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')

# Save the model
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)
