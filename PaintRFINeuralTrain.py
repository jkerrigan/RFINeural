#import matplotlib.pyplot as plt
import pylab as plt
from sklearn import mixture
from scipy import signal
import aipy as a
import numpy as n
from glob import glob
from sklearn.externals import joblib
from sklearn.neural_network import MLPClassifier
from matplotlib.patches import Rectangle


Plot=False
def loadAipyData(time):
#    HERAlist = glob('/Users/josh/Desktop/HERA/data/zen.2457458.'+str(time)+'*.xx.HH.uvcUA')
    HERAlist = glob('/Users/josh/Desktop/HERA/data/*A')
    HERAdata = []
    times = []
    for l in ['9_10']: #,'10_31','31_105','10_105','9_31','9_105','10_31']:
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

def corrPass(data1,data2):
    # correlate in time
    cCout = n.zeros_like(data1)
    for i in range(n.shape(data1)[1]):
        a = data1[:,i]#HERAdata[:,i] - n.mean(HERAdata[:,i])
        b = data2[:,i]#HERAdata[:,i] - n.mean(HERAdata[:,i])
        cCout[:,i] = signal.correlate(a,b,mode='same')
    return cCout

def expandData(data,mask,times,ct):
    sh = n.shape(data)
    expD = data
    nmask = mask
    ntimes = times
    for i in range(ct):
        newData = n.copy(data)
        newData += 0.01*(n.random.randn(sh[0],sh[1])+1j*n.random.randn(sh[0],sh[1]))
        expD = n.vstack((expD,newData))
        nmask = n.vstack((nmask,mask))
        ntimes = n.vstack((ntimes,times))
        del(newData)
    return expD,nmask,ntimes

def featArray(data,times):
    sh = n.shape(data)
    freqs = n.linspace(100,200,sh[1])
#    NNvar = n.zeros_like(data)
#    dvar = n.var(n.abs(data))
#    for i in range(sh[0]):
#        for j in range(sh[1]):
#            samples = []
#            for p in range(500):
#                k = n.random.randint(-1,1)
#                l = n.random.randint(-1,1)
#                try:
#                    samples.append(n.abs(data[k+i,l+j]))
#                except:
#                    pass
#            NNvar[i,j] = n.var(samples)
    corr = corrPass(data,data)
    corr = corrPass(corr,data)
    Corr = corr*n.conj(corr)

    X1 = n.zeros((sh[0]*sh[1],5))
    X1[:,0] = n.real(data).reshape(sh[0]*sh[1])
    X1[:,1] = n.imag(data).reshape(sh[0]*sh[1])
    X1[:,2] = (Corr.real).reshape(sh[0]*sh[1])
#    X1[:,2] = (n.log10(n.abs(NNvar)) - n.median(n.log10(n.abs(NNvar)))).reshape(sh[0]*sh[1])
    X1[:,3] = (n.array([freqs]*sh[0])).reshape(sh[0]*sh[1])
    X1[:,4] = (n.array([times]*sh[1])).reshape(sh[0]*sh[1])
    X1[n.abs(X1)>10**100] = 0
    for m in range(X1.shape[1]):
        X1[:,m] = X1[:,m]/n.abs(X1[:,m]).max()
    X1 = n.nan_to_num(X1)
    return X1

def subCluster(clust_sub):
    print 'Loading subcluster...'
    X = featArray(clust_sub)
    dpgmm = mixture.BayesianGaussianMixture(n_components=5,covariance_type='full',n_init=1,max_iter=1000,init_params='kmeans',weight_concentration_prior_type='dirichlet_process').fit(X)
    #dpgmm = mixture.GaussianMixture(n_components=5,covariance_type='full',n_init=1,max_iter=1000,init_params='kmeans').fit(X)                          
    labels = dpgmm.predict(X)
    labels = labels.reshape(-1,1024)
    return labels


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
from sklearn import cluster
data,times = loadAipyData(40)
print 'Data loaded. Preparing feature array...'
#X = featArray(data)
print 'X feature array loaded.'
#dpgmm = mixture.BayesianGaussianMixture(n_components=10,covariance_type='full',n_init=1,max_iter=1000,init_params='kmeans',weight_concentration_prior_type='dirichlet_process').fit(X)
#labels_array = []
#for i in range(6):
#    X = featArray(data[i,:,:])
#    print 'X feature array loaded.'
#    dpgmm = mixture.BayesianGaussianMixture(n_components=5,covariance_type='full',n_init=1,max_iter=1000,init_params='kmeans',weight_concentration_prior_type='dirichlet_process').fit(X)
    #dpgmm = mixture.GaussianMixture(n_components=5,covariance_type='full',n_init=1,max_iter=1000,init_params='kmeans').fit(X)
#    labels = dpgmm.predict(X)
#    labels = labels.reshape(-1,1024)
#    if i == 0:
#        labels_array = [labels]
#    else:
#        labels_array.append(labels)
#dpgmm = cluster.KMeans(n_clusters=30).fit(X)
#labels = dpgmm.labels_
plt.ion()
options={'y':-1,
    'n':100}

class Annotate(object):
    def __init__(self):
        self.ax = plt.gca()
#        self.ax = 
        self.rect = Rectangle((0,0), 1, 1)
        self.x0 = None
        self.y0 = None
        self.x1 = None
        self.y1 = None
        self.ax.add_patch(self.rect)
        self.ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.ax.figure.canvas.mpl_connect('button_release_event', self.on_release)

    def on_press(self, event):
        print 'press'
        self.x0 = event.xdata
        self.y0 = event.ydata

    def on_release(self, event):
        print 'release'
        self.x1 = event.xdata
        self.y1 = event.ydata
        self.rect.set_width(self.x1 - self.x0)
        self.rect.set_height(self.y1 - self.y0)
        self.rect.set_xy((self.x0, self.y0))
        self.ax.figure.canvas.draw()

data = data.reshape(-1,1024)

#mask = n.ones_like(data)
#print mask.shape
#fig = plt.figure()
#ax = fig.add_subplot(111)
class Refresh:
    def __init__(self,data,mask=None):
        self.fig = plt.figure()
        self.fig2 = plt.figure()
        if mask!=None:
            self.mask = mask
        else:
            self.mask = n.ones_like(data).astype(int)
        self.undomask = n.copy(self.mask)
        self.canvas = self.fig.canvas
        self.canvas2 = self.fig2.canvas
        self.ax = self.fig.gca()
        self.ax2 = self.fig2.gca()
#        self.ax = self.fig.add_subplot(111)
        #self.cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('button_press_event', self.undo)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.fig.canvas.mpl_connect('key_release_event', self.on_key_release)

        self.data = data
#        self.ax.subplot(211)
        self.ax.imshow(n.log10(n.abs(self.data))*n.abs(self.mask),aspect='auto',cmap='jet')
        self.ax.set_title('Visibility')
        self.ax.set_xlabel('Freq.')
        self.ax.set_ylabel('Time')
#        self.ax.subplot(212)
        self.ax2.imshow(n.log10(n.abs(self.delayTrans())),aspect='auto',cmap='jet')
        self.ax2.set_title('Delay Spectrum')
        self.ax2.set_xlabel('Delay Bin')
        self.ax2.set_ylabel('Time')
#        self.rect = Rectangle((0,0), 1, 1)
        self.x0 = None
        self.y0 = None
        self.x1 = None
        self.y1 = None
#        self.ax.add_patch(self.rect)
        #self.showVis()
    def on_key_press(self, event):
        if event.key == 'shift':
            self.shift_is_held = True
    def on_key_release(self, event):
        if event.key == 'shift':
            self.shift_is_held = False
    def onclick(self,event):
        print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' % (event.button, event.x, event.y, event.xdata, event.ydata))
        print self.mask[int(event.ydata),int(event.xdata)]
        self.undomask = self.mask
        self.mask[int(event.ydata)+1,int(event.xdata)+1] = 0
        print self.mask[int(event.ydata),int(event.xdata)]
        #self.ax.imshow(n.log10(n.abs(data))*n.abs(self.mask),aspect='auto',cmap='jet')
        #self.ax.imshow(n.log10(n.abs(data*self.mask)),aspect='auto',cmap='jet')
#        self.fig.canvas.draw()
    def undo(self, event):
        #print ('undo',event.button)
        if event.button == 3:
            print ('undo',event.button)
            print self.mask.sum()
            self.mask = self.undomask
            print self.mask.sum()
            self.ax.cla()
            self.ax.imshow(n.log10(n.abs(data))*n.abs(self.mask),aspect='auto',cmap='jet')
            self.canvas.draw()
            self.ax2.cla()
            self.ax2.imshow(n.log10(n.abs(self.delayTrans())),aspect='auto',cmap='jet')
            self.canvas2.draw()

    def on_press(self, event):
        if event.button == 1:
            print 'press'
            self.undomask = n.copy(self.mask)
            self.x0 = event.xdata
            self.y0 = event.ydata

    def on_release(self, event):
        if event.button == 1:
            print 'release'
            self.x1 = event.xdata
            self.y1 = event.ydata
        #self.rect.set_width(self.x1 - self.x0)
        #self.rect.set_height(self.y1 - self.y0)
        #self.rect.set_xy((self.x0, self.y0))
#            self.undomask = self.mask
#            self.fig.canvas.toolbar.push_current()
            self.mask[int(self.y0):int(self.y1)+1,int(self.x0):int(self.x1)+1] = 0
#        print self.mask[int(event.ydata),int(event.xdata)]
#            self.ax.cla()
            self.ax.imshow(n.log10(n.abs(self.data))*n.abs(self.mask),aspect='auto',cmap='jet')
            self.ax.set_title('Visibility')
            self.ax.set_xlabel('Freq.')
            self.ax.set_ylabel('Time')
            self.canvas.draw()
            self.ax2.cla()
            self.ax2.imshow(n.log10(n.abs(self.delayTrans())),aspect='auto',cmap='jet')
            self.ax2.set_title('Delay Spectrum')
            self.ax2.set_xlabel('Delay Bin')
            self.ax2.set_ylabel('Time')
            self.canvas2.draw()
    
    def gimmeMask(self):
        return self.mask

    def delayTrans(self):
        bh = a.dsp.gen_window(1024,window='blackman-harris')
        DATA = n.fft.fft(self.data*n.abs(self.mask)*bh,axis=1)
        DATA_ = n.fft.fftshift(DATA,axes=1)
        return DATA_
    #def showVis(self):
    #    self.ax.imshow(n.log10(n.abs(self.data*self.mask)),aspect='auto',cmap='jet')
    #    self.cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        #plt.pause(100000)
        #plt.show()
#cid = fig.canvas.mpl_connect('button_press_event',onclick)
#fig = plt.figure()
#ax = fig.add_subplot(111)
#ax.imshow(n.log10(n.abs(data*mask)),aspect='auto',cmap='jet')
#fig.canvas.draw()
msk = n.loadtxt('trainMask_severalBlines.txt')
refresher = Refresh(data,msk)

#refresher.showVis()
plt.pause(1000000)
plt.show()
mask = refresher.gimmeMask()
n.savetxt('trainMask_HQ.txt',mask.reshape(-1,1024))
#mask = n.loadtxt('trainMask.txt')
#ndata,nmask,ntimes = expandData(data,mask,times,2)
#XNN = featArray(ndata,ntimes)


plt.imshow(n.log10(n.abs(ndata*nmask)),aspect='auto',cmap='jet') 
plt.pause(100000)
plt.show()
#clf = MLPClassifier(batch_size=20,alpha=1e-4,hidden_layer_sizes=(510, 2),random_state=131,max_iter=10000)
#nmask = nmask.reshape(-1,1)
#clf.fit(XNN,nmask)

#predict_mask = clf.predict(XNN)
#print predict_mask.shape
#print clf.score(XNN,predict_mask)
#print (predict_mask.astype(bool)==nmask.astype(bool)).sum()/(nmask.astype(bool)).sum()
#predict_mask = predict_mask.reshape(-1,1024)
#plt.imshow(n.log10(n.abs(ndata*predict_mask)),aspect='auto',cmap='jet')
#plt.pause(100000)
#plt.show()

#data2,times = loadAipyData(50)
#data2 = data2.reshape(-1,1024)
#X2 = featArray(data2)
#sh2 = data2.shape
#mask2 = clf.predict(X2)
#mask2 = mask2.reshape(-1,1024)
#fig3 = plt.figure()
#plt.imshow(n.log10(n.abs(data2*mask2)),aspect='auto',cmap='jet')
#plt.pause(1000000)
#plt.show()
#plt.clf()

#joblib.dump(clf,'HERANeural.pkl')
