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

# Common use: 
#  dontrun=1  # Use this if we want to abort training behavior
#             # after functions are defined and Sh is initialized
#  exec(open("Shapes.py").read())

#############################################################
### class for 3D shape training
#############################################################

# Set up functions that allow shapes to be loaded, trained, and output
class Shapes:
    def __init__(self,sizex,sizey,sizez,verbose=True):
        self.sizex=sizex
        self.sizey=sizey
        self.sizez=sizez
        self.minval = 0.25   # Min non-zero voxel value for input (output)
        self.maxval = 0.599  # Max non-zero voxel value for input (output)
        self.minshift = self.minval + (1-self.maxval)  # Min value for image after shift to make maxval=1
        self.connectval = 1.0 # Initially, add strongly on connect values to aid training.
        self.useFeedback = True # Initially, use img2.csv data when available
        self.verbose=verbose
        if torch.cuda.is_available():
            self.dtype = torch.cuda.FloatTensor
        else:
            self.dtype = torch.FloatTensor
            
    def pltImages(self,image,name="plt") :  
        img=np.transpose(image.type(torch.FloatTensor).numpy(),(0,2,1,3))
        imageio.imwrite(name+".png",img.reshape(img.shape[0]*img.shape[1],img.shape[2]*img.shape[3]))

    # TIME NOTE: This routine can be slow but is not critical relative to training
    def prepIn(self) :
        sx = self.sizex
        sy = self.sizey
        sz = self.sizez
        # Initialize the image array
        ifile = np.loadtxt("img.csv",delimiter=",")
        self.images=int(ifile.shape[0])
        try:
            i2file = np.loadtxt("img2.csv",delimiter=",")
            self.images2=int(i2file.shape[0])
            self.Xraw = torch.cat((torch.Tensor(ifile.reshape(self.images,sz,sy,sx)),
                                   torch.Tensor(i2file.reshape(self.images2,sz,sy,sx))))
        except FileNotFoundError:
            self.Xraw = torch.Tensor(ifile.reshape(self.images,sz,sy,sx))
            self.images2=int(0)
        if self.useFeedback:
            self.imtot = int(self.images*16+self.images2)
        else:
            self.imtot = int(self.images*16)
        self.X = torch.Tensor(self.images+self.images2,sz,sy,sx).fill_(0).type(self.dtype)
        self.Xrnd = torch.Tensor(self.imtot,sz,sy,sx).fill_(0).type(self.dtype)
        rndp = torch.randperm(self.imtot)
        for i in range(self.images+self.images2):
            for z in range(sz): 
                for y in range(sy): 
                    for x in range(sx):
                        # This routine shifts maxval to 1 and allows for noise error on minval
                        val = self.Xraw[i][z][y][x]
                        val = 0 if val < self.minval*0.9 else min(1,max(self.minval,val)+(1-self.maxval))
                        self.X[i][z][y][x] = val
                        if i < self.images:
                            if y==0:
                                if x==0:
                                    val = self.X[i][z][0][0]
                                else:
                                    val = (self.X[i][z][0][x-1] + self.X[i][z][0][x])/2
                            else:
                                if x==0:
                                    val = (self.X[i][z][y-1][0] + self.X[i][z][y][0])/2
                                else:
                                    val = (self.X[i][z][y-1][x-1] + self.X[i][z][y-1][x] +
                                           self.X[i][z][y][x-1] + self.X[i][z][y][x])/4
                            self.Xrnd[rndp[i*16+8]][z][y][x] = val
            if i < self.images:
                for z in range(sz): 
                    for y in range(sy): 
                        for x in range(sx):
                            val = self.X[i][z][y][x]
                            self.Xrnd[rndp[i*16]][z][y][x] = val
                            # Reflections speed up generation step and emulate creating
                            #  more training samples from limited source data set.
                            self.Xrnd[rndp[i*16+1]][sz-z-1][y][x] = val
                            self.Xrnd[rndp[i*16+2]][z][sy-y-1][x] = val
                            self.Xrnd[rndp[i*16+3]][sz-z-1][sy-y-1][x] = val
                            self.Xrnd[rndp[i*16+4]][z][y][sx-x-1] = val
                            self.Xrnd[rndp[i*16+5]][sz-z-1][y][sx-x-1] = val
                            self.Xrnd[rndp[i*16+6]][z][sy-y-1][sx-x-1] = val
                            self.Xrnd[rndp[i*16+7]][sz-z-1][sy-y-1][sx-x-1] = val
                            # Shift image -0.5 in x and y
                            val = self.Xrnd[rndp[i*16+8]][z][y][x]
                            self.Xrnd[rndp[i*16+9]][sz-z-1][y][x] = val
                            self.Xrnd[rndp[i*16+10]][z][sy-y-1][x] = val
                            self.Xrnd[rndp[i*16+11]][sz-z-1][sy-y-1][x] = val
                            self.Xrnd[rndp[i*16+12]][z][y][sx-x-1] = val
                            self.Xrnd[rndp[i*16+13]][sz-z-1][y][sx-x-1] = val
                            self.Xrnd[rndp[i*16+14]][z][sy-y-1][sx-x-1] = val
                            self.Xrnd[rndp[i*16+15]][sz-z-1][sy-y-1][sx-x-1] = val
            elif self.useFeedback:
                for z in range(sz): 
                    for y in range(sy): 
                        for x in range(sx):
                            val = self.X[i][z][y][x]
                            swap = np.random.randint(1,8) # Always swap at least 1 dim
                            idx = rndp[self.images*15 + i]
                            self.Xrnd[idx][sz-z-1 if (swap & 1) else z][sy-y-1 if (swap & 2) else y][sx-x-1 if (swap & 4) else x] = val
    def buildnn(self) :
        self.net = nn.Sequential(
            nn.Linear(self.sizex*self.sizey*self.sizez,32),
            nn.Tanh(),
            nn.Linear(32,3),           # Bottleneck layer has 4 neurons
            nn.Tanh(),
            nn.Linear(3,64),
            nn.Tanh(),
            nn.Linear(64,1024),
            nn.Tanh(),
            nn.Linear(1024, self.sizex*self.sizey*self.sizez),
            nn.Sigmoid()
        ).type(self.dtype)
        # Report the "squared error" loss, not mean squared error
        # this gives a soft indication of the number of error
        # pixels during training, in addition to error amount
        self.loss_fn = torch.nn.MSELoss(size_average=False).type(self.dtype)
    def buildfeat(self) :
        self.feat = nn.Sequential(
            nn.Linear(self.sizex*self.sizey*self.sizez,32),
            nn.Tanh(),                 # Hyperbolic tangent for non-linearity
            nn.Linear(32,3),           # Bottleneck layer has 3 neurons
            nn.Tanh()
        ).type(self.dtype)
        netd = self.net.state_dict()
        featd = self.feat.state_dict()
        for i in range(2):
            featd[str(i*2)+'.weight'].copy_(netd[str(i*2)+'.weight'])
            featd[str(i*2)+'.bias'].copy_(netd[str(i*2)+'.bias'])
        self.feat.load_state_dict(featd)
        self.featout = self.feat(Variable(self.X.view(self.images+self.images2,self.sizex*self.sizey*self.sizez))).type(torch.FloatTensor)
    def buildgen(self) :
        self.gen = nn.Sequential(
            nn.Linear(3,64),           # 3 layer bottleneck is input 
            nn.Tanh(),
            nn.Linear(64,1024),     # Extra layers on generator based on autoencoder experiences       
            nn.Tanh(),
            nn.Linear(1024, self.sizex*self.sizey*self.sizez),
            nn.Sigmoid()
        ).type(self.dtype)
        netd = self.net.state_dict()
        gend = self.gen.state_dict()
        for i in range(3):
            gend[str(i*2)+'.weight'].copy_(netd[str(i*2+4)+'.weight'])
            gend[str(i*2)+'.bias'].copy_(netd[str(i*2+4)+'.bias'])
        self.gen.load_state_dict(gend)
    def loadgen(self):
        self.gen.load_state_dict(torch.load('gen.pt'))
    def adam(self,iterations,learningRate) :
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learningRate)
        bptr=int(0)
        for t in range(iterations):
            if (bptr+128 <= self.imtot):
                x = Variable(self.Xrnd[bptr:bptr+128].view(128,self.sizex*self.sizey*self.sizez)).type(self.dtype)
            else:
                x = Variable(torch.cat((self.Xrnd[bptr:self.imtot],self.Xrnd[0:(bptr+128)%self.imtot])).view(128,self.sizex*self.sizey*self.sizez)).type(self.dtype)
            bptr = (bptr+128)%self.imtot
            # Forward pass: compute predicted y by passing x to the model.
            y_pred = self.net(x).type(self.dtype)
        
            # Compute and print loss.
            loss = self.loss_fn(y_pred, x)  # Autoencoder uses input as expected output
            if self.verbose and t % 50 == 49:
                print("INFO: iteration "+str(t+1)+": "+str(loss.data[0]))
  
            # Before the backward pass, use the optimizer object to zero all of the
            # gradients for the variables it will update (which are the learnable weights
            # of the model)
            self.optimizer.zero_grad()

            # Backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()

            # Calling the step function on an Optimizer makes an update to its parameters
            self.optimizer.step()
    def mark(self,im,minshift,marked,z,y,x,component):
        if marked[z][y][x] :
            sys.exit("ERROR: component "+str(marked[z][y][x])+" already marked when starting "+str(component))
        # This function has too much recursion if called recursively to mark
        # all voxels, so implement as list traversal.  Every 3 elements is zyx
        voxels=list((z,y,x))
        while len(voxels) > 0:
            (z,y,x) = voxels[0:3]
            voxels = voxels[3:]
            if marked[z][y][x] :
                if marked[z][y][x] != component:
                    sys.exit("ERROR: Not expecting to see component "+str(marked[z][y][x])+" when marking "+str(component))
                continue
            marked[z][y][x] = component
            # Mark the 18-connected voxels near this one with same component
            if (x > 0 and im[z][y][x-1] > minshift*0.5): 
                voxels.extend((z,y,x-1))
            if (x < self.sizex-1 and im[z][y][x+1] > minshift*0.5): 
                voxels.extend((z,y,x+1))
            if (y > 0 and im[z][y-1][x] > minshift*0.5): 
                voxels.extend((z,y-1,x))
            if (y < self.sizey-1 and im[z][y+1][x] > minshift*0.5): 
                voxels.extend((z,y+1,x))
            if (z > 0 and im[z-1][y][x] > minshift*0.5): 
                voxels.extend((z-1,y,x))
            if (z < self.sizez-1 and im[z+1][y][x] > minshift*0.5): 
                voxels.extend((z+1,y,x))
            if (x > 0 and y > 0 and im[z][y-1][x-1] > minshift*0.5): 
                voxels.extend((z,y-1,x-1))
            if (x < self.sizex-1 and y > 0 and im[z][y-1][x+1] > minshift*0.5): 
                voxels.extend((z,y-1,x+1))
            if (x > 0 and y < self.sizey-1 and im[z][y+1][x-1] > minshift*0.5): 
                voxels.extend((z,y+1,x-1))
            if (x < self.sizex-1 and y < self.sizey-1 and im[z][y+1][x+1] > minshift*0.5): 
                voxels.extend((z,y+1,x+1))
            if (x > 0 and z > 0 and im[z-1][y][x-1] > minshift*0.5): 
                voxels.extend((z-1,y,x-1))
            if (x < self.sizex-1 and z > 0 and im[z-1][y][x+1] > minshift*0.5): 
                voxels.extend((z-1,y,x+1))
            if (x > 0 and z < self.sizez-1 and im[z+1][y][x-1] > minshift*0.5): 
                voxels.extend((z+1,y,x-1))
            if (x < self.sizex-1 and z < self.sizez-1 and im[z+1][y][x+1] > minshift*0.5): 
                voxels.extend((z+1,y,x+1))
            if (z > 0 and y > 0 and im[z-1][y-1][x] > minshift*0.5): 
                voxels.extend((z-1,y-1,x))
            if (z < self.sizez-1 and y > 0 and im[z+1][y-1][x] > minshift*0.5): 
                voxels.extend((z+1,y-1,x))
            if (z > 0 and y < self.sizey-1 and im[z-1][y+1][x] > minshift*0.5): 
                voxels.extend((z-1,y+1,x))
            if (z < self.sizez-1 and y < self.sizey-1 and im[z+1][y+1][x] > minshift*0.5): 
                voxels.extend((z+1,y+1,x))
    def checkandmark(self,im,minshift,newmarked,z,y,x,component):
        if not newmarked[z][y][x]:
            newmarked[z][y][x] = component
        elif newmarked[z][y][x] != component and np.random.uniform() < 0.1 + im[z][y][x]/minshift/5:
            im[z][y][x] = self.connectval
            return True
        return False
    # Grow 'marked' volume step by step until overlap encountered
    # Set intersection to valid some fraction of time (multiple
    # voxels likely to occur in same step, don't want too much
    # threadding added to image).  There are 3 growth steps, which
    # 1/3rd of components will hit each growth step:
    #  - Grow +/1 in x and +/-1 in y
    #  - Grow +/1 in x and +/-1 in y and +/1 in z
    #  - Grow to all 18-connected voxels (not the 8 corners of cube)
    def grow(self,im,minshift,marked,newmarked,z,y,x,pattern):
        component = marked[z][y][x]
        retval = False
        if not component :
            return retval  # No component to grow
        if (x > 0):
            retval |= self.checkandmark(im,minshift,newmarked,z,y,x-1,component)
        if (x < self.sizex-1):
            retval |= self.checkandmark(im,minshift,newmarked,z,y,x+1,component)
        if (y > 0):
            retval |= self.checkandmark(im,minshift,newmarked,z,y-1,x,component)
        if (y < self.sizey-1):
            retval |= self.checkandmark(im,minshift,newmarked,z,y+1,x,component)
        if (component%3 == (pattern+1)%3) or retval:
            return retval  # x and y are finer grid, always grow them
        if (z > 0):
            retval |= self.checkandmark(im,minshift,newmarked,z-1,y,x,component)
        if (z < self.sizez-1):
            retval |= self.checkandmark(im,minshift,newmarked,z+1,y,x,component)
        if (component%3 != pattern) or retval:
            return retval  # z grid is grown 2/3 of the time, 18-connected 1/3
        if (x > 0 and y > 0):
            retval |= self.checkandmark(im,minshift,newmarked,z,y-1,x-1,component)
        if (x < self.sizex-1 and y > 0):
            retval |= self.checkandmark(im,minshift,newmarked,z,y-1,x+1,component)
        if (x > 0 and y < self.sizey-1):
            retval |= self.checkandmark(im,minshift,newmarked,z,y+1,x-1,component)
        if (x < self.sizex-1 and y < self.sizey-1):
            retval |= self.checkandmark(im,minshift,newmarked,z,y+1,x+1,component)
        if retval:  # Return if x and y can connect (finer grid)
            return retval
        if (x > 0 and z > 0):
            retval |= self.checkandmark(im,minshift,newmarked,z-1,y,x-1,component)
        if (x < self.sizex-1 and z > 0):
            retval |= self.checkandmark(im,minshift,newmarked,z-1,y,x+1,component)
        if (x > 0 and z < self.sizez-1):
            retval |= self.checkandmark(im,minshift,newmarked,z+1,y,x-1,component)
        if (x < self.sizex-1 and z < self.sizez-1):
            retval |= self.checkandmark(im,minshift,newmarked,z+1,y,x+1,component)
        if (z > 0 and y > 0):
            retval |= self.checkandmark(im,minshift,newmarked,z-1,y-1,x,component)
        if (z < self.sizez-1 and y > 0):
            retval |= self.checkandmark(im,minshift,newmarked,z+1,y-1,x,component)
        if (z > 0 and y < self.sizey-1):
            retval |= self.checkandmark(im,minshift,newmarked,z-1,y+1,x,component)
        if (z < self.sizez-1 and y < self.sizey-1):
            retval |= self.checkandmark(im,minshift,newmarked,z+1,y+1,x,component)
        return retval
    def connect(self,im,minshift,marked,z,y,x):
        if marked[z][y][x] != 0 :
            sys.exit("ERROR: Connect called for marked location x,y,z = "+str(x)+","+str(y)+","+str(z))
        if im[z][y][x] > minshift*0.5 :
            sys.exit("ERROR: Connect called for pre-existing voxel location x,y,z = "+str(x)+","+str(y)+","+str(z))
        nearxh=0
        nearxm=0
        nearxl=0
        nearyh=0
        nearym=0
        nearyl=0
        nearzh=0
        nearzm=0
        nearzl=0
        # Mark the 18-connected voxels near this one with same component
        if (x > 0):
            nearxl |= 1<<marked[z][y][x-1]
            nearym |= 1<<marked[z][y][x-1]
            nearzm |= 1<<marked[z][y][x-1]
        if (x < self.sizex-1):
            nearxh |= 1<<marked[z][y][x+1]
            nearym |= 1<<marked[z][y][x+1]
            nearzm |= 1<<marked[z][y][x+1]
        if (y > 0):
            nearxm |= 1<<marked[z][y-1][x]
            nearyl |= 1<<marked[z][y-1][x]
            nearzm |= 1<<marked[z][y-1][x]
        if (y < self.sizey-1):
            nearxm |= 1<<marked[z][y+1][x]
            nearyh |= 1<<marked[z][y+1][x]
            nearzm |= 1<<marked[z][y+1][x]
        if (z > 0):
            nearxm |= 1<<marked[z-1][y][x]
            nearym |= 1<<marked[z-1][y][x]
            nearzl |= 1<<marked[z-1][y][x]
        if (z < self.sizez-1):
            nearxm |= 1<<marked[z+1][y][x]
            nearym |= 1<<marked[z+1][y][x]
            nearzh |= 1<<marked[z+1][y][x]
        if (x > 0 and y > 0):
            nearxl |= 1<<marked[z][y-1][x-1]
            nearyl |= 1<<marked[z][y-1][x-1]
            nearzm |= 1<<marked[z][y-1][x-1]
        if (x < self.sizex-1 and y > 0):
            nearxh |= 1<<marked[z][y-1][x+1]
            nearyl |= 1<<marked[z][y-1][x+1]
            nearzm |= 1<<marked[z][y-1][x+1]
        if (x > 0 and y < self.sizey-1):
            nearxl |= 1<<marked[z][y+1][x-1]
            nearyh |= 1<<marked[z][y+1][x-1]
            nearzm |= 1<<marked[z][y+1][x-1]
        if (x < self.sizex-1 and y < self.sizey-1):
            nearxh |= 1<<marked[z][y+1][x+1]
            nearyh |= 1<<marked[z][y+1][x+1]
            nearzm |= 1<<marked[z][y+1][x+1]
        if (x > 0 and z > 0):
            nearxl |= 1<<marked[z-1][y][x-1]
            nearym |= 1<<marked[z-1][y][x-1]
            nearzl |= 1<<marked[z-1][y][x-1]
        if (x < self.sizex-1 and z > 0):
            nearxh |= 1<<marked[z-1][y][x+1]
            nearym |= 1<<marked[z-1][y][x+1]
            nearzl |= 1<<marked[z-1][y][x+1]
        if (x > 0 and z < self.sizez-1):
            nearxl |= 1<<marked[z+1][y][x-1]
            nearym |= 1<<marked[z+1][y][x-1]
            nearzh |= 1<<marked[z+1][y][x-1]
        if (x < self.sizex-1 and z < self.sizez-1):
            nearxh |= 1<<marked[z+1][y][x+1]
            nearym |= 1<<marked[z+1][y][x+1]
            nearzh |= 1<<marked[z+1][y][x+1]
        if (z > 0 and y > 0):
            nearxm |= 1<<marked[z-1][y-1][x]
            nearyl |= 1<<marked[z-1][y-1][x]
            nearzl |= 1<<marked[z-1][y-1][x]
        if (z < self.sizez-1 and y > 0):
            nearxm |= 1<<marked[z+1][y-1][x]
            nearyl |= 1<<marked[z+1][y-1][x]
            nearzh |= 1<<marked[z+1][y-1][x]
        if (z > 0 and y < self.sizey-1):
            nearxm |= 1<<marked[z-1][y+1][x]
            nearyh |= 1<<marked[z-1][y+1][x]
            nearzl |= 1<<marked[z-1][y+1][x]
        if (z < self.sizez-1 and y < self.sizey-1):
            nearxm |= 1<<marked[z+1][y+1][x]
            nearyh |= 1<<marked[z+1][y+1][x]
            nearzh |= 1<<marked[z+1][y+1][x]
        # Clear out any 'marked=0' neighbors
        nearxh &= ~1
        nearxm &= ~1
        nearxl &= ~1
        nearyh &= ~1
        nearym &= ~1
        nearyl &= ~1
        nearzh &= ~1
        nearzm &= ~1
        nearzl &= ~1
        # Gather all nearby components
        nearby = nearxh | nearxm | nearxl
        if nearby==0:
            return 0   # No nearby components
        else :
            if ((nearby - 1) & nearby) == 0:  # Only one component encountered
                return 0
            elif ((nearxh!=0 and nearxm==0 and nearxl==0) or
                  (nearxl!=0 and nearxm==0 and nearxh==0) or
                  (nearyh!=0 and nearym==0 and nearyl==0) or
                  (nearyl!=0 and nearym==0 and nearyh==0) or
                  (nearzh!=0 and nearzm==0 and nearzl==0) or
                  (nearzl!=0 and nearzm==0 and nearzh==0)) :
                # This voxel sees a way to connect 2 components, but
                # there is a better voxel to use, so just return 0
                # (If 2 components are on same side of this voxel, then
                # the voxel one unit in that direction should be set
                # instead). This avoids creating too many connecter voxels.
                if self.verbose:
                    print("INFO: This should be covered by other voxel (nearby = "+str(nearby)+")")
                return 0
            else:
                # Set voxel strongly on to connect 2 or more components
                im[z][y][x] = self.connectval 
                return nearby   # Includes components connected
                
    # TIME NOTE: This routine can take a minute or so with 50 images
    def prepOut(self,rand=0,startimg=27,endimg=36) :
        sx = self.sizex;
        sy = self.sizey;
        sz = self.sizez;
        # Set the minimum and maximum values for the output images
        # Note: some of the code below assumes maxval > 1.5*minval
        # minval and maxval apply to final output image
        # A raw NN output from (minshift*0.5 up to infinity will get
        # mapped to be within [minval,maxval] range.
        minshift = self.minshift
        minval = self.minval
        maxval = self.maxval
        if rand==0:      # Generate outputs from X Sh.images (real training) data
            images=self.images
            X = torch.Tensor(images,self.sizez,self.sizey,self.sizex).fill_(0).type(self.dtype)
            for i in range(images):
                X[i]=self.X[i]
            predict=self.net(Variable(X.view(images,sx*sy*sz))).view(images,sz,sy,sx).type(torch.FloatTensor)
        elif rand <0:    # Generate images between X[startimg] and X[endimg]
            images=-rand
            X = torch.Tensor(images,3).fill_(0).type(self.dtype)
            start=self.featout[startimg]
            end=self.featout[endimg]
            for i in range(images):
                X[i]=(((end*i)+(start*(images-1-i)))/(images-1)).data
            predict=self.gen(Variable(X)).view(images,sz,sy,sx).type(torch.FloatTensor)
        else:            # Generate random images
            images=rand
            X = torch.rand(images,3).type(self.dtype)*2-1
            predict=self.gen(Variable(X)).view(images,sz,sy,sx).type(torch.FloatTensor)
        self.Xout = torch.Tensor(images,sz,self.sizey,self.sizex).fill_(0)
        numclean=0
        numsimple=0
        numinvert=0
        for i in range(images):
            self.Xout[i] = predict[i].data
            # Now insure we have a single connected nodule before pruning below minshift*0.5
            im=self.Xout[i]
            component=int(0)
            clean=True
            if ((im[0][0][0] > minshift*0.5 and im[sz-1][sy-1][sx-1] > minshift*0.5) or
                (im[0][0][sx-1] > minshift*0.5 and im[sz-1][sy-1][0] > minshift*0.5) or
                (im[0][sy-1][0] > minshift*0.5 and im[sz-1][0][sx-1] > minshift*0.5) or
                (im[sz-1][0][0] > minshift*0.5 and im[0][sy-1][sx-1] > minshift*0.5)):
                # Opposite corners set is not legal (inverted image)
                clean=False
                if self.verbose:
                    print("INFO: Xout["+str(i)+"] generated INVERTED")
                numinvert=numinvert+1
                im = 1.0-im
            while True:
                marked=np.zeros((sz,sy,sx),dtype=np.int)
                component=int(0)
                # Mark components with 18-connected groups
                for z in range(sz): 
                    for y in range(sy): 
                        for x in range(sx):
                            if (im[z][y][x] > minshift*0.5 and marked[z][y][x] == 0):
                                component += 1  # Number components starting with '1'
                                # self.mark will recursively mark all voxels that are 18-connected
                                self.mark(im,minshift,marked,z,y,x,component)
                if (component == 1):
                    if clean:
                        numclean=numclean+1
                        if self.verbose:
                            print("INFO: Xout["+str(i)+"] accepted clean as generated")
                    break  # Success!
                if self.verbose and clean:
                    print("INFO: Xout["+str(i)+"] has "+str(component)+" components ... connecting")
                if (component == 0):
                    # Shift voxels a bit stronger until we see components
                    im = im/minshift + minshift*0.05
                    continue
                # Add any voxels which can connect components and do a simple
                # connection tracking mask which can be queried to see if we are done
                mask = int(0)
                for z in range(sz): 
                    for y in range(sy): 
                        for x in range(sx):
                            if (marked[z][y][x] == 0):
                                # self.connect will set voxels that connect 2 components
                                new=self.connect(im,minshift,marked,z,y,x)
                                mask = mask if (new & mask == 0 and mask != 0) else new | mask
                # if there are 2 or 3 components, mask is all 1's if and only if
                # all components are now connected.  For 4 or more components, it is 
                # possible for them to be connected but not have 'mask' be all 1, so 
                # recheck connections if mask > 0
                if (mask == (2<<component) -2):
                    if clean:
                        numsimple=numsimple+1
                    break
                clean=False  # Neither a clean component nor simple reconnection
                if (mask > 0):
                    continue  # Some components were connected, check how many remain
                if (mask == 0):
                    break
                # Still haven't found connection, so grow image by finding voxels
                # that help connect components
                newvoxel = False
                growpattern = 0
                newmarked=np.copy(marked)
                while not newvoxel:
                    growpattern = (growpattern+1)%3   # Causes growth to proceed with relatively circular shape
                    for z in range(sz):
                        for y in range(sy):
                            for x in range(sx):
                                newvoxel |= self.grow(im,minshift,marked,newmarked,z,y,x,growpattern)
                    np.copyto(marked,newmarked)
                # End of connected loop - the grown image will now be checked for connetivity

            # Scale NN (and connection process) output from [minshift*0.5, infinity] down to [minval,maxval]
            for z in range(sz): 
                for y in range(sy): 
                    for x in range(sx):
                        val = im[z][y][x]
                        if (val < minshift*0.5):
                            self.Xout[i][z][y][x] = 0.0
                        elif (val < minshift):
                            self.Xout[i][z][y][x] = minval
                        elif (val < 1):
                            self.Xout[i][z][y][x] = val - (1-maxval)
                        else :
                            self.Xout[i][z][y][x] = maxval
        print("Images generated: "+str(images)+", clean: "+str(numclean)+
                      ", simple: "+str(numsimple)+", inverted: "+str(numinvert))

# Sh.pltImages(torch.stack((Sh.Xraw[23,6:14],Sh.Xraw[27,6:14],Sh.Xraw[32,6:14],
#               Sh.Xraw[36,6:14],Sh.Xraw[42,6:14],Sh.Xraw[46,6:14]))/0.599,"rawselected")

Sh=Shapes(40,40,20)
Sh.prepIn()
try:
  dontrun
except NameError:
  print("Starting network training")
else:
  raise ValueError("dontrun was set, stopped execution after processing input data")

Sh.buildnn()
Sh.adam(25000,0.0001)
Sh.prepOut()

Sh.buildfeat()
Sh.buildgen()
predict=Sh.net(Variable(Sh.X[0:Sh.images].view(Sh.images,32000))).view(Sh.images,20,40,40).type(torch.FloatTensor)
feat=Sh.feat(Variable(Sh.X[0:Sh.images].view(Sh.images,32000))).type(torch.FloatTensor)
gen=Sh.gen(feat.type(Sh.dtype)).view(Sh.images,20,40,40).type(torch.FloatTensor)
np.savetxt("bottleneck.csv",feat.data.numpy())
Xerr=((predict.data - Sh.X[0:Sh.images].cpu())**2).mean(dim=1).mean(dim=1).mean(dim=1)
np.savetxt("err.csv",Xerr.numpy(),fmt='%.6f')

Sh.prepOut(420)
np.savetxt("rnd.csv",Sh.Xout.numpy().reshape(420,40*40*20),delimiter=',',fmt='%.3f')
Sh.pltImages(Sh.Xout[0:6,6:14]/0.599,"rnd")
Sh.prepOut(-6)
Sh.pltImages(Sh.Xout[0:6,6:14]/0.599,"step")

