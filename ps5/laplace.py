import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import convolve
from scipy.interpolate import RegularGridInterpolator
from numba import njit
plt.style.use("ggplot")

##############
# Parameters #
##############
# Space size
l = 40
# Number of points along a given dimension
n = 400
# Grid spacing
d = l/n
# Conductor radius
r = 2
# Ring potential
Vr = 1
# Wall potential
Vw = 0
# Max Number of Steps
ns = 5000
# Threshold: Minimum summed potential change to consider convergence.
threshold = 0.01
##############
# Boundaries #
##############
# Given dimensions, coarseness and a radius,
# fills a mask array representing a ring of that radius
# in the given coarsened space
#@njit
def ring(n,d,r,c,mask):
    for i in range(n):
        for j in range(n):
            # If the line passes within 2d of a grid point, set that point to True
            # a little rough, but does a good job of forming a continuous ring.
            if np.abs((i*d - c)**2 + (j*d - c)**2 - r**2) <= 2*d:
                mask[i,j] = True

def disk(n,d,r,c,mask):
    for i in range(n):
        for j in range(n):
            # If the line passes within 2d of a grid point, set that point to True
            # a little rough, but does a good job of forming a continuous ring.
            if (i*d - c)**2 + (j*d - c)**2 <= r**2:
                mask[i,j] = True


def init(n,d,r,c):
    # The boundary conditions of the problem
    # And the mask that sets which grid cells to enforce
    bound = np.zeros((n,n))
    mask = np.zeros((n,n),dtype=bool)
    # Set the ring potential and mask
    disk(n,d,r,l/2,mask)
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
# Return the average around each point, with cyclic boundary conditions
# by rolling arrays and summing
#def mean(pot):
#    return (np.roll(pot,1,axis=0) + np.roll(pot,-1,axis=0) + 
#            np.roll(pot,1,axis=1) + np.roll(pot,-1,axis=1)) / 4

# Calculate average around each point by convolution.
# Proves to be slightly faster
def mean(pot):
    kernel = np.array([ [0.,0.25,0.],
                       [0.25,0.,0.25],
                        [0.,0.25,0.0]])
    return convolve(pot, kernel) 

# Calculate the laplacian of a potential
def laplace(pot, d=1):
    # Laplacian pot ~ mean(pot) - pot
    return (mean(pot) - pot) * 4 / d**2

# Calculate charge density from potential
def charge(pot):
    return -laplace(pot)

# Calculate total charge in and outside of ring
def in_out(chrg, d, n, r, c):
    mask = np.zeros((n,n), dtype=bool)
    ring(n,d,r,c,mask)
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
    bound,mask = init(n,d,r,l/2)
    # V is intially the boundary condition
    V = np.copy(bound)
    # At each step, update V and reset boundary conditions
    for i in range(ns):
        Vnew = mean(V)
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
    return V, mask

#V,mask = simple_laplace()
#chrg = charge(V)
#inside,outside = in_out(chrg,d,l,n,r,l/2)

#############################
# Part 2 Conjugate Gradient #
#############################
def Ax(pot,mask):
    Ax = charge(pot)
    Ax[mask]=0
    return Ax

def rhs(pot,mask):
    mat = mean(pot) * 4
    mat[mask]=0
    return mat

def conjgrad():
    # Initialize problem boundary and mask
    bound, mask = init(n,d,r,l/2)

    # Make rhs of conjugate gradient, becomes 
    b = rhs(bound,mask)
    V = 0 * bound
    res = b - Ax(V, mask)
    p = res.copy()
    for i in range(ns):
        Ap = Ax(p,mask)
        rtr = np.sum(res*res)
        print("iteration %d, resid is %f" % (i, rtr))
        alpha = rtr/np.sum(Ap * p)
        V = V + alpha*p
        rnew = res - alpha * Ap
        beta = np.sum(rnew * rnew)/rtr
        p = rnew + beta*p
        res = rnew
        if not (i%10):
            Vplot = np.copy(V)
            Vplot[mask] = bound[mask]
            plt.clf();
            plot_thing(Vplot)
            plt.pause(0.001)
        if rtr < threshold:
            print("Converged!")
            break
    else:
        print("Max Iterations Reached")
    return V, mask

#V,mask = conjgrad()
#chrg = charge(V)
#inside,outside = in_out(chrg,d,l,n,r,l/2)

####################################
# Part 3 Scaled Conjugate Gradient #
####################################
def upsample(pot,scale=2):
    s = pot.shape
    x,y = np.arange(s[0]), np.arange(s[1])
    interpolator = RegularGridInterpolator((x,y), pot)

    xnew = np.linspace(0, s[0]-1, s[0]*scale)
    ynew = np.linspace(0, s[1]-1, s[1]*scale)
    return interpolator(np.dstack([xnew,ynew]))
