#  exec(open("tail.py").read())
time.sleep(90)
while (os.path.getctime('rnd.csv') > os.path.getctime('feat.csv')):
  time.sleep(9)
time.sleep(2)
Sh.prepIn()

Xerr = torch.Tensor(Sh.images2,Sh.images).fill_(0).type(Sh.dtype)
if (Sh.images2 > 0):
  for i in range(Sh.images2):
    for j in range(Sh.images):
      Xerr[i,j]=((Sh.X[Sh.images+i] - Sh.X[j])**2).mean()
  np.savetxt("feederr.csv",Xerr.min(dim=1)[0].cpu().numpy(),fmt='%.6f')

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

