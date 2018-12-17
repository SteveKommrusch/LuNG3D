#  exec(open("final.py").read())
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
  Sh.pltImages(Sh.Xraw[Sh.images:Sh.images+6,6:14]/0.599,"final")
  np.savetxt("feederr.csv",Xerr.min(dim=1)[0].cpu().numpy(),fmt='%.6f')
