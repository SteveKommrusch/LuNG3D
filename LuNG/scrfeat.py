import time
import os.path
import sys
import imageio
import torch
import torch.nn as nn
import torch.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable

# load original 51 image CSV data
origfile = np.loadtxt("../LuNG/real/feat.csv",delimiter=",")
norig=int(origfile.shape[0])
Xorig = torch.Tensor(origfile[:,0:12])
if norig != 51:
  sys.exit("Error loading ../LuNG/real/feat.csv")

if len(sys.argv) == 1:
  sys.argv.append("real/feat420rnd.csv")

for idx in range(len(sys.argv)-1):
  cmpfile = np.loadtxt(sys.argv[idx+1],delimiter=",")
  # Test with: cmpfile = np.loadtxt("real/feat420rnd.csv",delimiter=",")
  ncmp=int(cmpfile.shape[0]-20)
  Xcmp = torch.Tensor(cmpfile[20:,0:12])
  if ncmp < 1:
    sys.exit("Error loading "+sys.argv[idx+1])
  
  Morig = Xorig.mean(0)
  Mcmp = Xcmp.mean(0)
  Sorig = Xorig.std(0,unbiased=False)
  Scmp = Xcmp.std(0,unbiased=False)
  Xerr = torch.Tensor(ncmp,norig).fill_(0)
  for i in range(ncmp):
    for j in range(norig):
      Xerr[i,j] = np.sqrt((((Xcmp[i] - Xorig[j])/Sorig)**2).sum())
  
  avgmindist = Xerr.min(1)[0].mean()
  mseofmeans = (((Mcmp-Morig)/Sorig)**2).mean()
  dirname=os.path.dirname(sys.argv[idx+1])
  f=open(dirname+"/../featdist.txt","a")
  f.write(dirname+": Average min feature dist = "+str(avgmindist)+", MSE of Means = "+str(mseofmeans)+"\n")
  f.close()
