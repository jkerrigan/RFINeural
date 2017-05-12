import numpy as n
import aipy as a
import pylab as pl
import xrfi
import torch
import torch.nn as nn
from torch.autograd import Variable
from glob import glob
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from sklearn.metrics import log_loss

def loadFullDay():
    HERAlist = glob('/users/jkerriga/data/jkerriga/HERA/zen.2457458.*.xx.HH.uvcUA')
    HERAdata = []
    times = []
    for k in HERAlist:
        uvHERA = a.miriad.UV(k)
        a.scripting.uv_selector(uvHERA, '9_10', 'xx')
        #temp = []
        for p,d,f in uvHERA.all(raw=True):
            #temp.append(d)
            HERAdata.append(d)
            times.append(uvHERA['lst'])
            #HERAdata.append(temp)
    HERAdata = n.array(HERAdata)
    times = n.array(times)
    return HERAdata,times

def injectRandomRFI(data,mask,injections):
    sh = n.shape(data)
    print sh
    outD = n.copy(data)
    outM = n.copy(mask)
    for i in range(injections):
        if n.abs(n.random.rand())>0.5:
            ## RFI across time                                                                                                                                             
            fw = n.random.randint(1,3)+1
            th = n.random.randint(1,900)+1
            fs = n.random.randint(1,sh[1]-fw)
            ts = n.random.randint(1,sh[0]-th)
            outD[ts:ts+th,fs:fs+fw] = outD[ts:ts+th,fs:fs+fw]+0.01*n.random.randn()*(n.random.randn(th,fw)+ 1j*n.random.randint(-1,2)*n.random.randn(th,fw))
            outM[ts:ts+th,fs:fs+fw] = 0.
        else:
            ## RFI across freq                                                                                                                 
            fw = n.random.randint(1,100)+1
            th = n.random.randint(1,3)+1
            fs = n.random.randint(1,sh[1]-fw)
            ts = n.random.randint(1,sh[0]-th)
            outD[ts:ts+th,fs:fs+fw] = outD[ts:ts+th,fs:fs+fw]+0.01*n.random.randn()*(n.random.randn(th,fw)+ 1j*n.random.randint(-1,2)*n.random.randn(th,fw))
            outM[ts:ts+th,fs:fs+fw] = 0.
        del(fw)
        del(th)
    return outD,outM


class CNN(nn.Module):
    def __init__(self,dropRate,kernel):
        self.dropRate = dropRate
        self.kernel = kernel
        super(CNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Conv2d(1, 16*1, kernel_size=self.kernel, padding=(self.kernel-1)/2),
            nn.BatchNorm2d(16*1),
            nn.MaxPool2d(kernel_size=(1,16)),                                                                                          
            nn.ReLU())

        self.layer2 = nn.Sequential(
            nn.Conv2d(1*16, 2*16, kernel_size=self.kernel, padding=(self.kernel-1)/2),
            nn.BatchNorm2d(2*16),                                                                                      
            nn.MaxPool2d(kernel_size=(2,1)),
            nn.ReLU())
        
        #### Dropout Layer ####
        self.layer3 = nn.Sequential(
            nn.Conv2d(2*16, 2*16, kernel_size=self.kernel, padding=(self.kernel-1)/2),
            nn.BatchNorm2d(2*16),
            nn.Dropout2d(p=self.dropRate),
            nn.ReLU())

        self.layer4 = nn.Sequential(
            nn.Linear(1024,1024),
            nn.ReLU())

        self.layer5 = nn.Sequential(
            nn.Linear(1024,1024),
            nn.ReLU())
        
        self.layer6 = nn.Sequential(
            nn.Linear(1024,1024),
            nn.ReLU())

        self.fc = nn.Linear(1024, 2*1024)

    def forward(self, x):
        out = self.layer1(x).view(1,16,-1,1024/16)
        out = self.layer2(out).view(1,2*16,-1,1024/16)                                                                               
        out = self.layer3(out).view(-1,1024).float()
        out = self.layer4(out).view(-1,1024).float()
        out = self.layer5(out).view(-1,1024).float()
        out = self.layer6(out).view(-1,1024).float()
        out = self.fc(out)
        return out

# Hyper Parameters                                                                                                                                                         
num_epochs = 1
batch_size = 1
grid = 5
lr = n.logspace(-5,-3,grid)
dr = n.linspace(0,0.9,grid)
ke = n.arange(1,11+grid,2)
LossGrid = n.zeros((grid,grid,grid))

###### Use a cost function to determine hyperparameters #####
for d in range(grid):
    print d
    for l in range(grid):
        for k in range(grid):
            DATA,times = loadFullDay()
            DATA = DATA[0:500,:]
            XRFImask = xrfi.xrfi_simple(DATA)
            mask = n.loadtxt('trainMask_HQ.txt')[0:500,:]*n.logical_not(XRFImask[0:500,:]).astype(int)
            #DATA,mask = injectRandomRFI(DATA,mask,int(injections))
            MASK1 = n.array(mask).reshape(10,-1,1024,1)
            DATA1 = DATA
            DATA1 = n.abs(DATA)
            DATA1 = DATA1/n.max(DATA1)
            DATA1 = DATA1.reshape(10,-1,1024,1)
            data1 = torch.from_numpy(DATA1)
            mask1 = torch.from_numpy(MASK1)
            train_dataset = TensorDataset(data1,mask1)
            train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
            test_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=False)
            cnn = CNN(dr[d],ke[k])
            criterion = nn.CrossEntropyLoss() 
            optimizer = torch.optim.Adam(cnn.parameters(), lr=lr[l])
    
            # Train the Model 
            for epoch in range(num_epochs):
                for i, (images, labels) in enumerate(train_loader):
                    images = Variable(images).float()
                    labels = Variable(labels[:,:,:,0]).long().view(-1)
                # Forwar   d + Backward + Optimize                                                                                                                                    
                    optimizer.zero_grad()
                    images = images.view(1,1,-1,1024)
                    outputs = cnn(images)                                                                                                                                       
                    outputs = outputs.view(-1,2)
                    loss = criterion(outputs.float(), labels.long())
                    loss.backward()
                    optimizer.step()
                    _, predicted = torch.max(outputs.data, 1)
            print 'Evaluating...'
            cnn.eval()
            ll = 0.
            for i, (images, labels) in enumerate(test_loader):
                images = Variable(images).float()
                images = images.view(1,1,-1,1024)
                outputs = cnn(images)
                outputs = outputs.view(-1,2)
                _, singPred = torch.max(outputs.data, 1)
                labels = labels.float().view(-1)
                ll += log_loss(labels.numpy(),singPred.numpy())
            LossGrid[d,l,k] = ll

bestFit = n.argwhere(LossGrid==n.min(LossGrid))
bF = bestFit[0]
print LossGrid[bF[0],bF[1],bF[2]],':','Best Drop Rate: ',dr[bF[0]],'Best Learning Rate: ',lr[bF[1]],'Best Kernel Size: ',ke[bF[2]]
