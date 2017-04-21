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
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(16),
            nn.Tanh())

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(3,3), padding=(1,1)),
            nn.Dropout(),
            nn.BatchNorm2d(32),
            nn.Tanh())

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(64),
            nn.Tanh())

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=(3,3), padding=(1,1)),
            nn.Dropout(),
            nn.BatchNorm2d(1),
            nn.Tanh())
        self.fc = nn.Linear(1024, 2*1024)
        
    def forward(self, x):
        out = self.layer1(x).view(1,16,-1,1024)
        out = self.layer2(out).view(1,32,-1,1024)
        out = self.layer3(out).view(1,64,-1,1024)
        print out.size()
        out = self.layer4(out)
        print out.size()
        out = out.view(-1, 1024).float()
        out = self.fc(out)
        return out
        


class TwoLayerNet(torch.nn.Module):
  def __init__(self, D_in, H1,H2,D_out):
    """                                                                                                                                                 
    In the constructor we instantiate two nn.Linear modules and assign them as                                                                          
    member variables.                                                                                                                                   
    """
    super(TwoLayerNet, self).__init__()
    self.linear1 = torch.nn.Linear(D_in, H1)
    self.linear2 = torch.nn.Linear(H1, H2)
    self.linear3 = torch.nn.Linear(H2, D_out)

  def forward(self, x):
    """                                                                                                                                                 
    In the forward function we accept a Variable of input data and we must return                                                                       
    a Variable of output data. We can use Modules defined in the constructor as                                                                         
    well as arbitrary operators on Variables.                                                                                                           
    """
    m = torch.nn.ReLU()
    p = torch.nn.ReLU()
    h1_relu = self.linear1(x)
    h2_relu = self.linear2(p(h1_relu))
    y_pred = self.linear3(m(h2_relu))
    return y_pred


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
for o in obs:
    print o
    uv = pyuvdata.miriad.Miriad()
    uv.read_miriad(o)
    for b in n.unique(uv.baseline_array):
        idx = uv.baseline_array==b
        data = uv.data_array[idx,0,:,0]
        data = n.abs(n.logical_not(uv.flag_array[idx,0,:,0])*data)
        data = medfilt(data)
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


