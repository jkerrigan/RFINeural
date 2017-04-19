import torch 
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as n
import aipy as a
import pylab as pl
from glob import glob

# Hyper Parameters
num_epochs = 1
batch_size = 256
learning_rate = 0.01

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
data,times = loadAipyData()
mask = n.loadtxt('trainMask_HQ.txt')
data = n.nan_to_num(data.reshape(-1,1024))
from torch.utils.data import DataLoader

# Create random Tensors to hold inputs and outputs, and wrap them in Variables                                                                          
from torch.utils.data import TensorDataset
#data = n.abs(data).astype(float)
data = n.array([data.real,data.imag])
mask = n.array([mask,mask])
mask = mask.reshape(-1,1024)
data = data.reshape(-1,1024)
#data = data/n.max(n.abs(data))
data1 = torch.from_numpy(data)
mask1 = torch.from_numpy(mask)
train_dataset = TensorDataset(data1,mask1)
train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
test_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=False)
del(mask1)
del(data1)
del(data)
# CNN Model (2 conv layer)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(2, 4*16*16, kernel_size=(3,25), padding=(1,12)),
            nn.BatchNorm2d(4*16*16),
            nn.Dropout(p=0.8),
            nn.ReLU(),
            nn.MaxPool2d((16,64)))
        self.layer2 = nn.Sequential(
            nn.Conv2d(4, 1, kernel_size=(5,51), padding=(2,25)),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.MaxPool2d(1))
        self.fc = nn.Linear(1024, 16*1024)
        
    def forward(self, x):
        out = self.layer1(x).view(1,4,-1,1024)
        print out.size()
        out = self.layer2(out)
        out = out.view(-1, 1024).float()
        out = self.fc(out)
        return out
        
cnn = CNN()


# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

# Train the Model
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
#        if i>9:
#            break
        images = Variable(images).float()
        labels = Variable(labels).long().view(-1)
        print images.size(),labels.size()
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        images = images.view(1,2,-1,1024)
        outputs = cnn(images)
        outputs = outputs.view(-1,2)
        print outputs.size(),labels.size()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        #if (i+1) % 100 == 0:
        print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data[0]))

# Test the Model
cnn.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
correct = 0
total = 0
for i,(images,labels) in enumerate(test_loader):
    images = Variable(images)
    images = images.view(1,2,-1,1024).float()
    outputs = cnn(images)
    outputs = outputs.view(-1,2)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    print predicted.size(),labels.size()
    correct += (predicted.int() == labels.int()).sum()

print('Test Accuracy of the model on the 10000 test images: %d %%' % (100 * correct / total))

predicted = predicted.view(-1,1024)
labels = labels.view(-1,1024)
pl.subplot(211)
pl.imshow(n.abs(labels.numpy()),aspect='auto')
pl.subplot(212)
pl.imshow(n.abs(predicted.numpy()),aspect='auto')
pl.show()
images = images.view(-1,1024).data.numpy()
pl.subplot(211)
pl.imshow(n.log10(n.abs(images))*n.abs(predicted.numpy()),aspect='auto',cmap='jet')
pl.subplot(212)
pl.imshow(n.log10(n.abs(images)),aspect='auto',cmap='jet')
pl.show()
# Save the Trained Model
torch.save(cnn, 'cnn.txt')
