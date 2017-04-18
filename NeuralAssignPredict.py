import os
import numpy as n
from scipy import signal
from sklearn import mixture
from sklearn import cluster
from scipy.signal import cwt
from scipy.stats import skewtest,kurtosistest
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from glob import glob
import pyuvdata
import pylab as pl
import optparse, sys, os
import aipy as a
from sklearn.externals import joblib
import torch
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

def featArrayPredict(data,times,NNarr):
    sh = n.shape(data)
    freqs = n.linspace(100,200,sh[1])
    X1 = n.zeros((sh[0]*sh[1],4))
    X1[:,0] = n.real(data).reshape(sh[0]*sh[1])
    X1[:,1] = n.imag(data).reshape(sh[0]*sh[1])
    X1[:,2] = NNarr #n.log10(n.abs(NNvar)).reshape(sh[0]*sh[1])
    X1[:,3] = (n.array([freqs]*sh[0])).reshape(sh[0]*sh[1])
    #X1[:,4] = (n.array([times]*sh[1])).reshape(sh[0]*sh[1])
    X1[n.abs(X1)>10**100] = 0
    for m in range(X1.shape[1]):
        X1[:,m] = X1[:,m]/n.abs(X1[:,m]).max()
    X1 = n.nan_to_num(X1)
    return X1

def normalize(X):
    normX = (X-n.mean(X))/n.std(X)
    return normX

def doStatTests(X,labels,ml):
    pca = PCA(n_components=1)
    pca.fit(X[labels==ml])
    XpcaML = pca.transform(X[labels==ml])
    labelsOut = labels
    normXpcaML = (XpcaML-n.mean(XpcaML))/n.std(XpcaML)
    #maxKurt = kurtosistest(normXpcaML)[1]
    #maxSkew = skewtest(normXpcaML)[1]
    for i in n.unique(labels):
        if len(X[labels==i])==0:
            continue
        else:
            Xpca = pca.transform(X[labels==i])
            Xpca = (Xpca-n.mean(Xpca))/n.std(Xpca)
            if len(Xpca) < 9:
                labelsOut[labels==i] = -1
                continue
            if False:
                if len(Xpca) < 9:
                    labelsOut[labels==i] = -1
                    continue
                pl.figure()
                if skewtest(Xpca)[1] > 0.5 or kurtosistest(Xpca)[1] > 0.5:
                    tag = 'RFI'
                else:
                    tag = 'Not RFI'
                sk = skewtest(Xpca)[1]
                kt = kurtosistest(Xpca)[1]
                sk1 = skewtest(XpcaML)[1]
                kt1 = kurtosistest(XpcaML)[1]
                pl.subplot(211)
                pl.hist(Xpca,50,label=tag+':'+str(sk)+':'+str(kt))
                pl.legend()
                pl.subplot(212)
                pl.hist(XpcaML,50,label=tag+':'+str(sk1)+':'+str(kt1))
                pl.legend()
                pl.show()
            if i == ml:
                continue
            if skewtest(Xpca)[1] > 0.01: #or kurtosistest(Xpca)[1] > 1.:
                labelsOut[labels==i] = -1
            #else:
            #    labelsOut[labels==i] = ml
    return labelsOut

def LabelMaker(newlabels):
    ml1 = 0
    ml1num = 0
    for t in n.unique(newlabels):
        if (newlabels==t).sum()>ml1num:
            ml1 = t
            ml1num = (newlabels==t).sum()
    newlabels[newlabels==ml1] = -1
    newlabels[newlabels!=-1] += 10
    return newlabels

def findMaxLabel(labels):
    mlabel = ()
    maxCt = 0
    for i in n.unique(labels):
        if (labels==i).sum() > maxCt:
            maxCt = (labels==i).sum()
            mlabel = i
        else:
            continue
    return mlabel

def findMinLabel(labels):
    mlabel = ()
    minCt = 10**100
    for i in n.unique(labels):
        if (labels==i).sum() < minCt:
            minCt = (labels==i).sum()
            mlabel = i
        else:
            continue
    return mlabel



o = optparse.OptionParser()
opts,obs = o.parse_args(sys.argv[1:])

#clf = joblib.load('HERANeural.pkl')
#RFIlabels = n.loadtxt('RFIlabels.txt')
#RFIlabels = n.round(RFIlabels,0)
CL1 = []
CL2 = []
CL3 = []
model = torch.load('heranet.txt')
for o in obs:
    print o
    #try:
    uv = pyuvdata.miriad.Miriad()
    uv.read_miriad(o)
    #except:
    #    pass
    for b in n.unique(uv.baseline_array):
        idx = uv.baseline_array==b
        data = uv.data_array[idx,0,:,0]
        data = n.abs(n.logical_not(uv.flag_array[idx,0,:,0])*data)
        data1 = torch.Tensor(data/data.max())
        data1V = Variable(data1)
        sh = n.shape(data)
        #times = uv.lst_array[idx]
        #X = featArray(data,times)
        maskV = model(data1V)
        mask = n.round(maskV.data.numpy())
        #mask = clf.predict(X)
        #mask = mask.reshape(sh[0],sh[1])
        #for i in RFIlabels:
        #    labels[labels==i] = -1
        #labels[labels!=-1] = 1
        #ml = findMaxLabel(labels)
        #print 'Max label:',ml
        #mask = n.zeros_like(data).astype(bool)
        #mask[labels!=ml] = True
        mask = n.logical_not(mask.astype(bool))
        uv.flag_array[idx,0,:,0] = mask
        del(mask)
#        del(X)
    uv.write_miriad(o+'r')
    del(uv)


