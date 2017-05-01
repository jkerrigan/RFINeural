import os
import numpy as n
from glob import glob
import pyuvdata
import pylab as pl
import optparse, sys, os
import aipy as a
from xrfi import xrfi_simple

o = optparse.OptionParser()
opts,obs = o.parse_args(sys.argv[1:])

for o in obs:
    print o
    uv = pyuvdata.miriad.Miriad()
    uv.read_miriad(o)
    for b in n.unique(uv.baseline_array):
        idx = uv.baseline_array==b
        data = uv.data_array[idx,0,:,0]
        xrfi_mask = xrfi_simple(data)
        sh = n.shape(data)
        uv.flag_array[idx,0,:,0] = xrfi_mask
        del(xrfi_mask)
    uv.write_miriad(o+'X')
    del(uv)


