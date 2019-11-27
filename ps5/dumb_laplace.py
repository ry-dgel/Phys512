import numpy as np
from matplotlib import pyplot as plt
plt.ion()

n=400

V=np.zeros([n,n])
bc=0*V

mask=np.zeros([n,n],dtype='bool')
mask[:,0]=True
mask[:,-1]=True
mask[0,:]=True
mask[-1,:]=True
#mask[n//2,n//4:(3*n)//4]=True
mask[n//4:n//2,n//4:n//2]=True
bc[n//4:n//2,n//4:n//2]=1.0
mask[n//4:n//2,n//2:(3*n)//4]=True
bc[n//4:n//2,n//2:(3*n)//4]=-1.0


#bc[n//2,n//4:(3*n)//4]=1

V=bc.copy()

for i in range(2*n):
    V[1:-1,1:-1]=(V[1:-1,0:-2]+V[1:-1,2:]+V[:-2,1:-1]+V[2:,1:-1])/4.0
    V[mask]=bc[mask]
    #plt.clf()
    #plt.imshow(V)
    #plt.colorbar()
    #plt.pause(0.001)
rho=V[1:-1,1:-1]-(V[1:-1,0:-2]+V[1:-1,2:]+V[:-2,1:-1]+V[2:,1:-1])/4.0
