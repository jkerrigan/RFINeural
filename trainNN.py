import torch 
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import aipy as a
import pylab as pl
from glob import glob
import numpy as n

# Hyper Parameters
num_epochs = 5
batch_size = 100
learning_rate = 0.001

def loadAipyData():
    HERAlist = glob('/Users/josh/Desktop/HERA/data/*A')
    HERAdata = []
    times = []
    for l in ['9_10']: #,'31_105','10_105','9_31','9_105','10_31']:                                                                                    
        data = []
        for k in HERAlist:
            uvHERA = a.miriad.UV(k)
            a.scripting.uv_selector(uvHERA, l, 'xx')
            for p,d,f in uvHERA.all(raw=True):
                data.append(d)
                times.append(uvHERA['lst'])
        if l == '9_10':
            HERAdata = [data]
        else:
            HERAdata.append(data)
    print n.shape(HERAdata)
    HERAdata = n.array(HERAdata)
    times = n.array(times)
    return HERAdata,times


# MNIST Dataset
train_dataset = dsets.MNIST(root='../data/',
                            train=True, 
                            transform=transforms.ToTensor(),
                            download=True)

test_dataset = dsets.MNIST(root='../data/',
                           train=False, 
                           transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, 
                                          shuffle=False)

# CNN Model (1 conv layer)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16*4, kernel_size=(193,769), padding=0),
            nn.ReLU())
            #nn.MaxPool2d(5))
            #nn.BatchNorm2d(16),
#            nn.ReLU(),
#            nn.MaxPool2d(2))
#        self.layer2 = nn.Sequential(
#            nn.Conv2d(16, 32, kernel_size=5, padding=2),
#            nn.BatchNorm2d(32),
#            nn.ReLU(),
#            nn.MaxPool2d(2))
        self.fc = nn.Linear(1024*256, 2) ##size of input images *2
        
    def forward(self, x):
        out = self.layer1(x)
        print out.size()
#        out = self.layer2(out)
        out = out.view(-1,1024)
        out = self.fc(out)
        return out
        
cnn = CNN()


# Loss and Optimizer
#criterion = nn.CrossEntropyLoss(weight=None,)
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
data,time = loadAipyData()
mask = n.loadtxt('trainMask_severalBlines.txt')[0:256,:]
data = n.nan_to_num(data.reshape(-1,1024))
data = n.array(n.abs(data[0:256,:]))
rfiFlags = Variable(torch.Tensor(mask).int())
criterion = nn.CrossEntropyLoss(weight=None,size_average=True)
# Train the Model
for epoch in range(num_epochs):
    #for i, (images, labels) in enumerate(train_loader):
   #     print images.size()
   #     print labels.size()
    vis = Variable(torch.Tensor(data.reshape(1,1,-1,1024)))
#    rfiFlags = Variable(torch.Tensor(mask))
        
        # Forward + Backward + Optimize
    optimizer.zero_grad()
    outputs = cnn(vis)
        #outputs = outputs.transpose(0,1)
        #outputs = outputs.contiguous().view(1,-1).int()
    #rfiFlags = rfiFlags.view(1,-1)
    loss = criterion(outputs, rfiFlags)
#        loss.backward()
#        optimizer.step()
        
#        if (i+1) % 100 == 0:
#            print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' 
#                   %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data[0]))

# Test the Model
cnn.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
correct = 0
total = 0
for images, labels in test_loader:
    images = Variable(images)
    outputs = cnn(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

print('Test Accuracy of the model on the 10000 test images: %d %%' % (100 * correct / total))

# Save the Trained Model
torch.save(cnn.state_dict(), 'cnn.pkl')
