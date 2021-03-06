import os
import numpy as n
from scipy.signal import medfilt

from glob import glob
import pyuvdata
import pylab as pl
import optparse, sys, os
import aipy as a
from sklearn.externals import joblib
import torch
import torch.nn as nn
from torch.autograd import Variable
def loadFullDay():
    HERAlist = glob('/Users/josh/Desktop/HERA/data/zen.2457458.*.xx.HH.uvcUA')
    HERAdata = []
    times = []
    for k in HERAlist:
        uvHERA = a.miriad.UV(k)
        a.scripting.uv_selector(uvHERA, '9_10', 'xx')
        for p,d,f in uvHERA.all(raw=True):
            HERAdata.append(d)
            times.append(uvHERA['lst'])
    HERAdata = n.array(HERAdata)
    times = n.array(times)
    return HERAdata,times

def localStats(data,k,l):
    samples = []
    for p in range(100):
        i = n.random.randint(-1,1)
        j = n.random.randint(-1,1)
        try:
            samples.append(n.abs(data[k+i,l+j]))
        except:
            pass
    return n.var(samples)

class CNN(nn.Module):
    def __init__(self,dropRate):
        self.dropRate = dropRate
        super(CNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Conv2d(1, 16*1, kernel_size=5, padding=2),
            nn.BatchNorm2d(16*1),
            nn.MaxPool2d(kernel_size=(1,16)),
            #nn.Dropout(p=self.dropRate),
            nn.ReLU())

        self.layer2 = nn.Sequential(
            nn.Conv2d(1*16, 2*16, kernel_size=5, padding=2),
            nn.BatchNorm2d(2*16),
            #nn.Dropout2d(p=self.dropRate),
            nn.MaxPool2d(kernel_size=(2,1)),
            nn.ReLU())

        self.layer3 = nn.Sequential(
            nn.Conv2d(2*16, 2*32, kernel_size=5, padding=2),
            nn.BatchNorm2d(2*32),
            #nn.MaxPool2d(kernel_size=(1,2)),
            nn.ReLU())

        self.layer4 = nn.Sequential(
            nn.Conv2d(2*32, 2*16, kernel_size=5, padding=2),
            #nn.Dropout2d(p=self.dropRate),
            nn.BatchNorm2d(2*16),
            nn.ReLU())

        self.layer5 = nn.Sequential(
            nn.Conv2d(2*8*2, 2*8*2, kernel_size=5, padding=2),
            nn.BatchNorm2d(2*8*2),
            nn.ReLU())
        
        self.layer6 = nn.Sequential(
            nn.Conv2d(2*16, 2*16, kernel_size=5, padding=2),
            nn.BatchNorm2d(2*16),
            nn.Dropout2d(p=self.dropRate),
            nn.ReLU())
        
        self.layer6_5 = nn.Sequential(
            nn.Linear(1024,1024),
            nn.ReLU())

        self.layer7 = nn.Sequential(
            nn.Linear(1024,1024),
            nn.ReLU())

        self.layer8 = nn.Sequential(
            nn.Linear(1024,1024),
            nn.ReLU())
        
        self.fc = nn.Linear(1024, 2*1024)

    def forward(self, x):
        out = self.layer1(x).view(1,16,-1,1024/16)
        out = self.layer2(out).view(1,2*16,-1,1024/16)
        out = self.layer3(out).view(1,2*32,-1,1024/16)
        out = self.layer4(out).view(1,2*16,-1,1024/16)
        out = self.layer5(out).view(1,2*16,-1,1024/16)
        out = self.layer6(out).view(-1,1024).float()
        out = self.layer6_5(out).view(-1,1024).float()
        out = self.layer7(out).view(-1,1024).float()
        out = self.layer8(out).view(-1,1024).float()
        
        out = self.fc(out)
        return out

def corrPass(data1,data2):
    cCout = n.zeros_like(data1)
    for i in range(n.shape(data1)[1]):
        a = data1[:,i]
        b = data2[:,i]
        cCout[:,i] = signal.correlate(a,b,mode='same')
    return cCout

def featArray(data,times):
    sh = n.shape(data)
    freqs = n.linspace(100,200,sh[1])
    corr = corrPass(data,data)
    corr = corrPass(data,corr)
    Corr = corr*n.conj(corr)
    X1 = n.zeros((sh[0]*sh[1],5))
    X1[:,0] = n.real(data).reshape(sh[0]*sh[1])
    X1[:,1] = n.imag(data).reshape(sh[0]*sh[1])
    #X1[:,2] = (n.log10(n.abs(NNvar)) - n.median(n.log10(n.abs(NNvar)))).reshape(sh[0]*sh[1])
    X1[:,2] = (Corr.real).reshape(sh[0]*sh[1])
    X1[:,3] = (n.array([freqs]*sh[0])).reshape(sh[0]*sh[1])
    X1[:,4] = (n.array([times]*sh[1])).reshape(sh[0]*sh[1])
    X1[n.abs(X1)>10**100] = 0
    for m in range(X1.shape[1]):
        X1[:,m] = X1[:,m]/n.abs(X1[:,m]).max()
    X1 = n.nan_to_num(X1)
    #X1 = normalize(X1,norm='l2',axis=0)
    return X1

def normalize(X):
    normX = (X-n.mean(X))/n.std(X)
    return normX

o = optparse.OptionParser()
opts,obs = o.parse_args(sys.argv[1:])

#clf = joblib.load('HERANeural.pkl')
#RFIlabels = n.loadtxt('RFIlabels.txt')
#RFIlabels = n.round(RFIlabels,0)
CL1 = []
CL2 = []
CL3 = []
#model = torch.load('heranet.txt')
cnn = torch.load('DEEPcnn.txt')
cnn.eval()
for o in obs:
    print o
    uv = pyuvdata.miriad.Miriad()
    uv.read_miriad(o)
    for b in n.unique(uv.baseline_array):
        idx = uv.baseline_array==b
        data = uv.data_array[idx,0,:,0]
        data = n.abs(n.logical_not(uv.flag_array[idx,0,:,0])*data)
        #data = n.abs((data-n.median(data))/n.std(data))
        data = data/n.max(data)
        data1 = torch.Tensor(data)
        data1V = Variable(data1)
        data1V = data1V.view(1,1,-1,1024)
        sh = n.shape(data)
        #times = uv.lst_array[idx]
        #X = featArray(data,times)
        maskV = cnn(data1V)
        maskV = maskV.view(-1,2)
        _, predicted = torch.max(maskV.data, 1)
        maskPred = n.round(predicted.numpy())
        print n.shape(maskPred)
        maskPred = maskPred.reshape(sh[0],sh[1])
        #mask = clf.predict(X)
        #mask = mask.reshape(sh[0],sh[1])
        #for i in RFIlabels:
        #    labels[labels==i] = -1
        #labels[labels!=-1] = 1
        #ml = findMaxLabel(labels)
        #print 'Max label:',ml
        #mask = n.zeros_like(data).astype(bool)
        #mask[labels!=ml] = True
        maskPred = n.logical_not(maskPred.astype(bool))
        uv.flag_array[idx,0,:,0] = maskPred
        del(maskPred)
#        del(X)
    uv.write_miriad(o+'r')
    del(uv)


