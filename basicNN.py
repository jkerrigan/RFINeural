import torch
from torch.autograd import Variable
from glob import glob
import aipy as a
import numpy as n
import pylab as pl
"""
A fully-connected ReLU network with one hidden layer, trained to predict y from x
by minimizing squared Euclidean distance.
This implementation defines the model as a custom Module subclass. Whenever you
want a model more complex than a simple sequence of existing Modules you will
need to define your model this way.
"""
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

def loadAipyDataTest():
    HERAlist = glob('/Users/josh/Desktop/HERA/data/*U')
    HERAdata = []
    times = []
    for l in ['9_31']: 
#,'31_105','10_105','9_31','9_105','10_31']:                                                                                    
        data = []
        for k in HERAlist:
            uvHERA = a.miriad.UV(k)
            a.scripting.uv_selector(uvHERA, l, 'xx')
            for p,d,f in uvHERA.all(raw=True):
                data.append(d)
                times.append(uvHERA['lst'])
        if l == '9_31':
            HERAdata = [data]
        else:
            HERAdata.append(data)
    print n.shape(HERAdata)
    HERAdata = n.array(HERAdata)
    times = n.array(times)
    return HERAdata,times

def expandMask(mask,batch):
    sh = n.shape(mask)
#    expData = data
    expMask = mask
    for i in range(batch):
#        expData = n.vstack((expData,data+0.1*n.random.randn()*(n.random.randn(sh[0],sh[1])+1j*n.random.randn(sh[0],sh[1]))))
        expMask = n.vstack((expMask,mask))
    return expMask

data,time = loadAipyData()
mask = n.loadtxt('trainMask_HQ.txt')
data = n.nan_to_num(data.reshape(-1,1024))
mask = mask

#mask = expandMask(mask,5)
#print n.shape(data)
#print n.shape(mask)

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
    #maybe add a dropout layer? see what that does
  def forward(self, x):
    """
    In the forward function we accept a Variable of input data and we must return
    a Variable of output data. We can use Modules defined in the constructor as
    well as arbitrary operators on Variables.
    """
    drop = torch.nn.Dropout(p=0.9)
    m = torch.nn.ReLU()
    p = torch.nn.ReLU()
    h1_relu = self.linear1(x)
    h2_relu = self.linear2(p(drop(h1_relu)))
    y_pred = self.linear3(m(h2_relu))
    return y_pred


# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H1,H2, D_out = 390,1024,1024,1024,1024
epochs = 100
from torch.utils.data import DataLoader

# Create random Tensors to hold inputs and outputs, and wrap them in Variables
from torch.utils.data import TensorDataset
data = n.abs(data).astype(float)
data = data/n.max(n.abs(data))
#print n.shape(data)
data1 = torch.from_numpy(data)
mask1 = torch.from_numpy(mask)
Container = TensorDataset(data1,mask1)

#train_data = DataLoader(Container,batch_size=N,shuffle=True)
#x = Variable(torch.Tensor(n.abs(data[0:1000,:])))
#y = Variable(torch.Tensor(mask[0:1000,:].astype(int)), requires_grad=False).view(-1)

# Construct our model by instantiating the class defined above
model = TwoLayerNet(D_in, H1,H2,D_out)

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = torch.nn.MSELoss()
#criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adagrad(model.parameters(), lr=1e-2)
#for t in range(1000):
lossVal = 10000000.
t = 0
lossArray = []
accuracy = []
#while lossVal>800.: 
for t in range(epochs):
    train_data = DataLoader(Container,batch_size=N,shuffle=True)
    for i,j in enumerate(train_data):
  # Forward pass: Compute predicted y by passing x to the model
        x = Variable(j[0].float())
        y = Variable(j[1].float())
#        print x.mean()
#        print y.mean()
        y_pred = model(x)

#        y_pred = y_pred.view(-1,2)
        y_pred = y_pred.float()
        y = y.view(-1,1024)
#        print y_pred.size(),y.size()
  #print y_pred.data.numpy()
  # Compute and print loss
        loss = criterion(y_pred, y)
        ypred = n.array(n.round(y_pred.data.numpy()))
        ynump = n.array(n.round(y.data.numpy()))
        shp = n.shape(ynump)

        print (n.sum(ynump==ypred)).astype(float)/(shp[0]*shp[1])
        acc = (n.sum(ynump==ypred)).astype(float)/(shp[0]*shp[1])
#  print acc,'%'
#  accuracy.append(acc)
#  loss = log_loss(y_pred,y)
        print(t, loss.data[0])
        accuracy.append(acc)
  # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    lossVal = loss.data[0]
    lossArray.append(lossVal)
#y_pred = y_pred[:,0]
#pred = y_pred[:,1].contiguous()
pl.subplot(211)
pl.plot(n.log10(n.array(lossArray)))
pl.subplot(212)
pl.plot(n.array(accuracy))
pl.show()

pl.subplot(211)
pl.imshow(n.abs(n.round(mask)),aspect='auto')
pl.subplot(212)
pl.imshow(n.abs(n.round(y_pred.view(-1,1024).data.numpy())),aspect='auto')
pl.show()


data_test,times_test = loadAipyDataTest()
data_test = data_test/n.abs(data_test).max()
x1 = Variable(torch.Tensor(n.abs(data_test[0,:,:])))
torch.save(model,'heranet.txt')
mask_pred = model(x1)
def delayspectrum(dat):
    DATA = n.fft.fft(dat,axis=1)
    DATA_ = n.fft.fftshift(DATA,axes=1)
    return DATA_
rfiMasked = n.round(mask_pred.data.numpy())
dataMask = data_test[0,:,:]*rfiMasked
bh = a.dsp.gen_window(1024,window='blackman-harris')
pl.subplot(211)
pl.imshow(n.log10(n.abs(dataMask)),aspect='auto',cmap='jet')
pl.subplot(212)
pl.imshow(n.log10(n.abs(delayspectrum(bh*dataMask))),aspect='auto',cmap='jet')
pl.show()
