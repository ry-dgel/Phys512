import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import convolve
from scipy.interpolate import RegularGridInterpolator
plt.style.use("ggplot")

##############
# Parameters #
##############
# Space size
l = 40
# Number of points along a given dimension
n = 256
# Grid spacing
d = l/n
# Conductor radius
r = 8
# Ring potential
Vr = 1
# Wall potential
Vw = 0
# Max Number of Steps
ns = 50000
# Threshold for methods
threshold = 0.01

##############
# Boundaries #
##############
# Given dimensions, coarseness a radius, and a center
# fills a mask array representing a disk of that radius centered on center
# in the given coarsened space
def disk(n,d,r,c,mask):
    for i in range(n):
        for j in range(n):
            # Fill in points that are within r of the center defined by c.
            if (i*d - c[0])**2 + (j*d - c[1])**2 <= r**2:
                mask[i,j] = True

# Make a box of a fixed potential and associated mask 
def box(n):
    bound = np.zeros((n,n))
    mask = np.zeros((n,n),dtype=bool)

    # Set the wall potential and mask
    for edge in [mask[0,:],mask[-1,:],mask[:,0],mask[:,-1]]:
        edge[:] = True
    for edge in [bound[0,:],bound[-1,:],bound[:,0],bound[:,-1]]:
        edge[:] = Vw

    return bound,mask

def init(n,d,r,c):
    # The boundary conditions of the problem
    # And the mask that sets which grid cells to enforce
    bound = np.zeros((n,n))
    mask = np.zeros((n,n),dtype=bool)
    # Set the ring potential and mask
    disk(n,d,r,c,mask)
    bound[mask] = Vr
    # Set the wall potential and mask
    for edge in [mask[0,:],mask[-1,:],mask[:,0],mask[:,-1]]:
        edge[:] = True
    for edge in [bound[0,:],bound[-1,:],bound[:,0],bound[:,-1]]:
        edge[:] = Vw
    return bound, mask

# Plotting Functions
def plot_thing(thing):
    max_abs = np.max(np.abs(thing))
    fig = plt.imshow(thing, cmap=plt.cm.twilight, 
                     vmin=-max_abs, vmax=max_abs)
    plt.colorbar()
    return fig

def ax_thing(thing,ax,cb=True):
    max_abs = np.max(np.abs(thing))
    im = ax.imshow(thing, cmap=plt.cm.twilight,
                   vmin=-max_abs, vmax=max_abs)
    return im

####################
# Part 0 Analytics #
####################
def an_pot(n,d,r,c):
    bound,mask = init(n,d,r,c)
    W = l/2
    Q = Vr/(2*np.log(W/r))
    V = lambda x,y: 2*Q*np.log(W/np.sqrt(x**2+y**2))
    for i in range(n):
        for j in range(n):
            if not mask[i,j]:
                bound[i,j] = V((i-n//2)*d, (j-n//2)*d) 
    return bound

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
def rho(pot):
    return -laplace(pot)

# Calculate abs value of gradient of potential
def field(pot):
    x,y = np.gradient(pot)
    abs = np.sqrt(x**2+y**2)
    return abs

def simple_laplace(bound,mask):
    # V is intially the boundary condition
    V = np.copy(bound)
    # At each step, update V and reset boundary conditions
    for i in range(ns):
        Vnew = mean(V)
        Vnew[mask] = bound[mask]
        delta = np.sum(Vnew-V)
        V = Vnew
        print("Itartion %d, delta = %f" % (i, delta))
        if not (i%40):
            plt.clf();
            plot_thing(V)
            plt.pause(0.001)
        if delta < threshold:
            print("Converged!")
            break
    else:
        print("Max Iterations Reached")

    return V

# Initialize
bound,mask = init(n,d,r,[l/2, l/2])
# Run Sim
V = simple_laplace(bound,mask)
# Calc Charge Density
chrg = rho(V)
density = np.sum(chrg[1:-1,1:-1])/(2*np.pi*r)
print("Charge density is: %f" % density)


# Calculate analytic solution
anV = an_pot(n,d,r,[l/2,l/2])
# Plot all results
fig, axes = plt.subplots(2,2)
plt.colorbar(ax_thing(V,axes[0,0]),ax=axes[0,0])
axes[0,0].set_title("Simulated Potential")
plt.colorbar(ax_thing(chrg,axes[0,1]), ax=axes[0,1])
axes[0,1].set_title("Simulated Charge Distribution")
plt.colorbar(ax_thing(anV,axes[1,0]), ax=axes[1,0])
axes[1,0].set_title("Analytic Potential")
plt.colorbar(ax_thing(V-anV, axes[1,1]), ax=axes[1,1])
axes[1,1].set_title("Simulated - Analytic")
plt.show(block=False)
plt.pause(1)

#############################
# Part 2 Conjugate Gradient #
#############################
def Ax(pot,mask):
    pot[mask]=0
    Ax = rho(pot)
    Ax[mask]=0
    return Ax

def Ax2(mat,mask,copy=True):
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

def rhs(pot,mask):
    pot[np.invert(mask)]=0
    mat = mean(pot) * 4
    mat[mask]=0
    return mat

def conj_step(V,p,res,mask):
    Ap = Ax(p,mask)
    rtr = np.sum(res*res)
    alpha = rtr/np.sum(Ap*p)
    V = V+alpha*p
    rnew = res - alpha * Ap
    beta = np.sum(rnew*rnew)/rtr
    p = rnew + beta*p
    res = rnew
    return V,p,res

def conjgrad(bound,mask,threshold,V=None,plot=True):
    b = rhs(bound,mask)
    if V is None:
        V = 0 * bound
    res = b - Ax(V, mask)
    p = res.copy()
    Vplot = 0 * V
    i=0 # Keep for returning number of iterations
    for i in range(ns):
        V,p,res = conj_step(V,p,res,mask)
        rtr = np.sum(res*res)
        print("iteration %d, resid is %f" % (i, rtr))
        if not (i%10) and plot:
            Vplot = np.copy(V)
            Vplot[mask] = bound[mask]
            plt.clf();
            plot_thing(Vplot)
            plt.pause(1)
        if rtr < threshold:
            print("Converged!")
            break
    else:
        print("Max Iterations Reached")
    V[mask] = bound[mask]
    return V, i+1

# Initialize
bound, mask = init(n,d,r,[l/2,l/2])
# Run Sim
V,_ = conjgrad(bound, mask, threshold)

# Plot Results
plt.clf()
plot_thing(V)
plt.show(block=False)
plt.pause(1)
####################################
# Part 3 Scaled Conjugate Gradient #
####################################
# Simple interpolation upsampler. Using scipy
def upsample(mat,scale=2):
    s = mat.shape
    # Generate the coordinate grid
    x,y = np.arange(s[0]), np.arange(s[1])
    # Make the interpolator object using x,y and the given matrix
    interpolator = RegularGridInterpolator((x,y), mat)
    # Make a function that interpolates on a scaled grid
    f = lambda i,j: interpolator(np.dstack([i/np.max(i)*np.max(x), 
                                            j/np.max(j)*np.max(y)]))
    # Return data on scaled grid
    return np.fromfunction(f,[int(round(scale*d)) for d in s])

# Effectively taken from example code. Couldn't get more clever
# methods such as average or convolving to work.
def deres(mat,scale=2):
    mm=np.zeros([mat.shape[0]//scale,mat.shape[1]//scale],dtype=mat.dtype)
    mm=np.maximum(mm,mat[::scale,::scale])
    mm=np.maximum(mm,mat[::scale,1::scale])
    mm=np.maximum(mm,mat[1::scale,::scale])
    mm=np.maximum(mm,mat[1::scale,1::scale])
    return mm

def scaling_conjgrad(bound,mask,passes,threshold):
    # Descale boundary condition and mask
    masks = [mask]
    bounds = [bound]
    for i in range(1,passes):
        masks.append(deres(masks[i-1]))
        bounds.append(deres(bounds[i-1]))
    masks.reverse()
    bounds.reverse()
    
    tol = threshold * 2**(passes-1)
    V = 0*masks[0]
    total_iters = 0
    for i in range(passes):
        print("Solving pot with downscale factor: %d" % 2**(passes-i-1))
        bc = bounds[i]
        ms = masks[i]
        # Conjgrad solve V at new scale
        V,iters = conjgrad(bc,ms,tol,V,plot=True) 
        Vplot = np.copy(V)
        Vplot[ms] = bc[ms]
        total_iters += iters
        if i < passes-1:
            # Upscale potential for next round, want to average with
            # boundary conditions considered, so use the Vplot which
            # had these added.
            V = upsample(Vplot)
        tol /= 2

    V[mask] = bound[mask]
    print("Total iterations = %d" % total_iters)
    return V

# Initialize
bound, mask = init(n,d,r,[l/2,l/2])
# Run Sim
V = scaling_conjgrad(bound,mask,6,threshold)#
# Plot Results
plt.clf()
plot_thing(V)
plt.show(block=False)
plt.pause(1)

###############
# Part 4 Bump #
###############
# Initialize the main disk and bump segment
bound, mask = init(n,d,r,[l/2,l/2])
bb, mb = init(n,d,r/10,[l/2 - r,l/2])
# Add bump to main disk
mask[mb] = mb[mb]
bound[mb] = bb[mb]

# Run simulation
V = scaling_conjgrad(bound,mask,6,threshold)
# Plot results
plt.clf()
plot_thing(field(V))
plt.show(block=False)
plt.pause(1)

###############
# Part 5 Temp #
###############
# Initialize Box
bound, mask = box(n)
# Temp sweep parameters
tmax = 1
tstep = 0.01
Tramp = 10 * tstep
temp = 0 * bound
lines = []

# Lazy "Adiabatic"
for t in range(int(round(tmax/tstep))):
    # Update wall temp
    bound[0,:] = (t+1) * Tramp
    # Run CG
    temp,_ = conjgrad(bound,mask,threshold,temp,plot=False)
    # Save line from wall to middle
    line = temp[:,l//2]
    lines.append(line) 
   
plt.clf()
plt.figure()
for line in lines:
    plt.plot(line)
plt.show(block=False)
plt.pause(1)
