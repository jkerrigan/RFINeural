import torch 
import xrfi
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as n
import aipy as a
import pylab as pl
from glob import glob
from scipy import signal
from scipy.signal import medfilt

# Hyper Parameters
num_epochs = 20
batch_size = 1
learning_rate = 10**(-3)

def loadAipyData():
    HERAlist = glob('/Users/josh/Desktop/HERA/data/*A')
    HERAlist = n.sort(HERAlist)
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
    #print n.shape(HERAdata)
    HERAdata = n.array(HERAdata)
    times = n.array(times)
    return HERAdata,times

def expandMask(data,mask,batch):
    sh = n.shape(mask)
    expData = data
    expMask = mask
    for i in range(batch):
        expData = n.vstack((expData,data+0.001*n.random.randn()*(n.random.randn(sh[0],sh[1])+ 1j*n.random.randint(-1,2)*n.random.randn(sh[0],sh[1]))))
        expMask = n.vstack((expMask,mask))
    return expData,expMask

def injectRandomRFI(data,mask,injections):
    sh = n.shape(data)
    for i in range(injections):
        if n.abs(n.random.rand())>0.5:
            ## RFI across time
            fw = n.random.randint(1,4)
            th = n.random.randint(1,900)
            fs = n.random.randint(1,sh[1]-fw)
            ts = n.random.randint(1,sh[0]-th)
            data[ts:ts+th,fs:fs+fw] = data[ts:ts+th,fs:fs+fw]+0.001*n.random.randn()*(n.random.randn(th,fw)+ 1j*n.random.randint(-1,2)*n.random.randn(th,fw))
            mask[ts:ts+th,fs:fs+fw] = 0.
        else:
            ## RFI across freq
            fw = n.random.randint(1,100)
            th = n.random.randint(1,4)
            fs = n.random.randint(1,sh[1]-fw)
            ts = n.random.randint(1,sh[0]-th)
            data[ts:ts+th,fs:fs+fw] = data[ts:ts+th,fs:fs+fw]+0.001*n.random.randn()*(n.random.randn(th,fw)+ 1j*n.random.randint(-1,2)*n.random.randn(th,fw))
            mask[ts:ts+th,fs:fs+fw] = 0.
            
    return data,mask

def corrPass(data1,data2):
    cCout = n.zeros_like(data1)
    for i in range(n.shape(data1)[1]):
        a = data1[:,i]
        b = data2[:,i]
        cCout[:,i] = signal.correlate(a,b,mode='same')
    return cCout

data,times = loadAipyData()
data = data.reshape(-1,1024)
XRFImask = xrfi.xrfi_simple(data)
data = data[0:3900,:]

mask = n.loadtxt('trainMask_HQ.txt')[0:3900,:]*n.logical_not(XRFImask[0:3900,:]).astype(int) 
data = n.nan_to_num(data.reshape(-1,1024))
from torch.utils.data import DataLoader
sh = n.shape(data)
print n.shape(data),n.shape(mask)
# Create random Tensors to hold inputs and outputs, and wrap them in Variables  
data,mask = expandMask(data,mask,1)                                       
from torch.utils.data import TensorDataset
#data = n.vstack((data,expDat))
#mask = n.vstack((mask,expMask))
#data,mask = injectRandomRFI(data,mask,300)

data = n.abs(data)
data = (data-n.median(data))/n.max(data)
pl.subplot(211)
pl.imshow(n.log10(data),aspect='auto',interpolation='none')
pl.subplot(212)
pl.imshow(n.log10(data*mask),aspect='auto',interpolation='none')
pl.show()
#data = n.array([n.abs(data),n.angle(data)])
#mask = n.array([mask,mask])
mask = mask.reshape(78,-1,1024)
data = data.reshape(78,-1,1024)
#print n.shape(data)
#data = (n.abs(data))
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
    def __init__(self,dropRate):
        self.dropRate = dropRate
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Conv2d(1, 2, kernel_size=5, padding=2),
            nn.BatchNorm2d(2),
            #nn.Dropout(p=self.dropRate),
            nn.Tanh())

        self.layer2 = nn.Sequential(
            nn.Conv2d(2, 2, kernel_size=11, padding=5),
            nn.BatchNorm2d(2),
            nn.Tanh())

        self.layer3 = nn.Sequential(
            nn.Conv2d(2, 2, kernel_size=11, padding=5),
            nn.BatchNorm2d(2),
            nn.Dropout(p=self.dropRate),
            nn.Tanh())
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=5, padding=2),
            nn.BatchNorm2d(1),
            nn.Tanh())

        self.layer5 = nn.Sequential(
            nn.Linear(1024,1024),
            nn.Tanh())
        self.fc = nn.Linear(1024, 2*1024)

    def forward(self, x):
        out = self.layer1(x).view(1,2,-1,1024)
        out = self.layer2(out).view(1,2,-1,1024)
        out = self.layer3(out).view(1,2,-1,1024)
        out = self.layer4(out).view(-1,1024).float()
        out = self.layer5(out).view(-1,1024).float()
        out = self.fc(out)
        return out
        
cnn = CNN(0.8)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
loss_array = []
correct = []
# Train the Model
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images).float()
        labels = Variable(labels).long().view(-1)
        print images.size(),labels.size()
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        images = images.view(1,1,-1,1024)
        outputs = cnn(images)
        outputs = outputs.view(-1,2)
        loss = criterion(outputs.float(), labels.long())
        loss.backward()
        optimizer.step()
        loss_array.append(loss.data[0])
        _, predicted = torch.max(outputs.data, 1)
        correct.append(1.0*(predicted.int()==labels.data.int()).sum()/labels.size()[0])
        #if (i+1) % 100 == 0:
        print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' %(epoch+1, num_epochs, i+1, len(train_dataset)/batch_size, loss.data[0]))
torch.save(cnn, 'DEEPcnn.txt')
# Test the Model
cnn.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
correct_test = 0
total = 0
ct_array = []
ct_totarray = []
for i,(images,labels) in enumerate(test_loader):
    images = Variable(images)
    images = images.view(1,1,-1,1024).float()
    outputs = cnn(images)
    outputs = outputs.view(-1,2)
    _, predicted = torch.max(outputs.data, 1)
    total += predicted.size(0)
    ct_totarray.append(total)
    correct_test += 1.0*(predicted.int() == labels.int()).sum()
    ct_array.append(correct_test)

print n.array(ct_array)/n.array(ct_totarray)
pl.figure()
pl.subplot(211)
pl.plot(loss_array)
pl.subplot(212)
pl.plot(correct)
pl.plot(n.array(ct_array)/n.array(ct_totarray),'r')
pl.savefig('losscorr.png')

#print('Test Accuracy of the model on the 10000 test images: %d %%' % (100 * correct / total))
pl.figure()
predicted = predicted.view(-1,1024)
labels = labels.view(-1,1024)
pl.subplot(211)
pl.imshow(n.abs(labels.numpy()),aspect='auto')
pl.subplot(212)
pl.imshow(n.abs(n.round(predicted.numpy())),aspect='auto')
pl.show()
images = images.view(-1,1024).data.numpy()
pl.subplot(211)
pl.imshow(n.log10(n.abs(images*n.round(predicted.numpy()))),aspect='auto',cmap='jet')
pl.subplot(212)
pl.imshow(n.log10(n.abs(images)),aspect='auto',cmap='jet')
pl.show()
# Save the Trained Model
#torch.save(cnn, 'DEEPcnn.txt')
