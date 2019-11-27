import numpy as np
import matplotlib.pyplot as plt
from numba import njit
plt.style.use("ggplot")

##############
# Parameters #
##############
# Space size
l = 10
# Number of points along a given dimension
n = 100
# Grid spacing
d = l/n
# Ring radius
r = 2
# Ring potential
Vr = 1
# Wall potential
Vw = 0
# Max Number of Steps
ns = 5000
# Threshold: Minimum summed potential change to consider convergence.
threshold = 1E-5 * n**2
##############
# Boundaries #
##############
# Given dimensions, coarseness and a radius,
# fills a mask array representing a ring of that radius
# in the given coarsened space
#@njit
def ring(l,n,d,r,c,mask):
    for i in range(n):
        for j in range(n):
            # If the line passes within 2d of a grid point, set that point to True
            # a little rough, but does a good job of forming a continuous ring.
            if np.abs((i*d - c)**2 + (j*d - c)**2 - r**2) <= 2*d:
                mask[i,j] = True

def init(l,n,d,r,c):
    # The boundary conditions of the problem
    # And the mask that sets which grid cells to enforce
    bound = np.zeros((n,n))
    mask = np.zeros((n,n),dtype=bool)
    # Set the ring potential and mask
    ring(l,n,d,r,l/2,mask)
    bound[mask] = Vr
    # Set the wall potential and mask
    for edge in [mask[0,:],mask[-1,:],mask[:,0],mask[:,-1]]:
        edge[:] = True
    for edge in [bound[0,:],bound[-1,:],bound[:,0],bound[:,-1]]:
        edge[:] = Vw
    return bound, mask

##########################
# Part 1 Simple and Lazy #
##########################
# Calculate the laplacian of a potential
def laplace(pot):
    # Average around each point by rolling arrays and summing
    mean = (np.roll(pot,1,axis=0) + np.roll(pot,-1,axis=0) + 
            np.roll(pot,1,axis=1) + np.roll(pot,-1,axis=1)) / 4
    # Laplacian pot ~ mean(pot) - pot
    return mean - pot

# Calculate charge density from potential
def charge(pot):
    return -laplace(pot)

# Calculate total charge in and outside of ring
def in_out(chrg, d, l, n, r, c):
    mask = np.zeros((n,n), dtype=bool)
    ring(l,n,d,r,c,mask)
    inside = np.sum(chrg[mask])
    outside = np.sum(chrg[np.invert(mask)])
    return inside, outside

# Plotting Functions
def plot_thing(thing):
    max_abs = np.max(np.abs(thing))
    fig = plt.imshow(thing, cmap=plt.cm.twilight, 
                     vmin=-max_abs, vmax=max_abs)
    plt.colorbar()
    return fig

def simple_laplace():
    bound,mask = init(l,n,d,r,l/2)
    # V is intially the boundary condition
    V = np.copy(bound)
    # At each step, update V and reset boundary conditions
    for i in range(ns):
        Vnew = V + laplace(V)
        Vnew[mask] = bound[mask]
        delta = np.sum(Vnew-V)
        V = Vnew
        print("Itartion %d, delta = %f" % (i, delta))
        if not (i%10):
            plt.clf();
            plot_thing(V)
            plt.pause(0.001)
        if delta < threshold:
            print("Converged!")
            break
    else:
        print("Max Iterations Reached")
    return V

#V = simple_laplace()
#chrg = charge(V)
#inside,outside = in_out(chrg,d,l,n,r,l/2)

#############################
# Part 2 Conjugate Gradient #
#############################
def Ax(V,mask):
    Vuse = laplace(V) - V
    Vuse[mask] = 0
    return Vuse


def ax_2d(mat,mask,copy=False):
    """Write the Laplacian operator in the way we need it to be.  Note that the boundary conditions as specified by the mask
    do not enter into the matrix since they are on the right-hand side of the matrix equation.  So, set them to zero here, then we won't have
    to worry about handling them separately."""
    if copy:
        mat=mat.copy()
    mat[mask]=0
    mm=4*mat
    mm[:,:-1]=mm[:,:-1]-mat[:,1:]
    mm[:,1:]=mm[:,1:]-mat[:,:-1]
    mm[1:,:]=mm[1:,:]-mat[:-1,:]
    mm[:-1,:]=mm[:-1,:]-mat[1:,:]
    mm[mask]=0
    return mm

bound, mask = init(l,n,d,r,l/2)
b = -laplace(bound)
V = 0 * bound
res = b - ax_2d(V, mask)
p = res.copy()
for i in range(ns):
    Ap = ax_2d(p,mask)
    rtr = np.sum(res*res)
    print("iteration %d, resid is %f" % (i, rtr))
    alpha = rtr/np.sum(Ap * p)
    V = V + alpha*Ap
    rnew = res - alpha * Ap
    beta = np.sum(rnew * rnew)/rtr
    p = rnew + beta*p
    res = rnew
    if not (i%10):
        Vplot = np.copy(V)
        Vplot[mask] = bound
        plt.clf();
        plot_thing(Vplot)
        plt.pause(0.001)
    if rtr < threshold:
        print("Converged!")
        break
else:
    print("Max Iterations Reached")

