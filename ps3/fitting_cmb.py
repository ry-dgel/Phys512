import numpy as np
import camb
from matplotlib import pyplot as plt
from datetime import datetime
plt.style.use("ggplot")

##########
# Params #
##########
steps = 7000
# Initital parameters
params=np.asarray([65,0.02,0.1,0.05,2e-9,0.96])


########
# Data #
########
# Load the data, extract 'x' axis
wmap=np.loadtxt('wmap_tt_spectrum_9yr_v5.txt')
ls = wmap[:,0]

#############
# Functions #
#############
# Write the data by appending it to the given file.
def write(path, params, chisq, accept):
    with open(path, "a+") as f:
        for param in params:
            f.write("%e," % param)
        f.write("%f," % chisq)
        f.write("%.2f\n" % accept)
    print("wrote to file")

# Generate the spectrum for a set of parameters and ls via camb.
# Mostly copied from given example code.
# This part contains a memory leak somewhere...
def get_spectrum(ls,pars):
    H0=pars[0]
    ombh2=pars[1]
    omch2=pars[2]
    tau=pars[3]
    As=pars[4]
    ns=pars[5]

    pars=camb.CAMBparams()
    pars.set_cosmology(H0=H0,ombh2=ombh2,omch2=omch2,mnu=0.06,omk=0,tau=tau)
    pars.InitPower.set_params(As=As,ns=ns,r=0)
    pars.set_for_lmax(int(max(ls)),lens_potential_accuracy=0)

    results=camb.get_results(pars) # Memory leaked on this line??
    powers=results.get_cmb_power_spectra(pars,CMB_unit='muK')
    cmb=powers['total']
    tt=cmb[:,0]
    del results
    del cmb
    del pars
    return tt[ls.astype(int)]

# Get the gradient around a set of parameters, with small displacement ds.
# each element of ds is of the form (index, delta)
# Where index gives the parameter index and delta the amount by which that
# parameter should be varied.
def get_grad(ls, pars, ds):
    # Generate output matrix
    grad = np.zeros((len(ls),len(ds)))
    
    # i = index of output, idx = index of gradient, d = delta of parameter
    for i,pair in enumerate(ds):
        # Get values
        idx = pair[0]
        d = pair[1]
        
        # Generate vector [0,...,0,ds,0,...0] to add/sub to parameter vector
        delta = np.zeros(len(pars))
        delta[idx] = d
        
        # Calculate the spectrum at two points, and calculate derivative
        plus = get_spectrum(ls, pars + delta)
        mins = get_spectrum(ls, pars - delta)
        # Set output for this index
        grad[:,i] = (plus - mins)/(2 * d)

    return grad

# Calculte residuals, studentized residuals and chi square.
def resid(true, fit):
    return true - fit

def studentized(true, fit, error):
    return resid(true,fit)/error

def chisqr(true,fit,error):
    return np.sum(np.square(studentized(true,fit,error)))

# Determine optimal ds size by scaling and plotting
# 70 Seems like a decent scaling for the ds
"""
steps = 10
means = np.zeros((steps,1))
for i, scale in enumerate(np.linspace(5,200,steps)):
    ds = list(zip([2], np.array([0.1])/scale))
    grad = get_grad(ls, params, ds)
    print(grad[:,0])
    plt.plot(grad[:,0], label=str(scale))
plt.legend()
plt.show(block=True)
"""

############
# Plotting #
############
# Little wrappers for plotting the output function with residuals
def plot_res(ax, x, stud_res):
    ax.errorbar(x, stud_res, 1, fmt='o', markersize=3)

def plot_dat(ax, x, y, e):
    ax.errorbar(x,y,e, fmt='o', markersize=3)

def plot_func(ax, x, f, label=""):
    ax.plot(x,f, zorder=1000, label=label)

#######################
# Levenberg-Marquardt #
#######################
# A Levenberg-Marquardt fitting routine
# f = func to fit, function of x,p
# g = grad of f, function of x,p
# x,y,err = input data, x-values, y-values and errors on measurement
# guess = initial parameter guess
# max_iter = maximum number of steps
# rel_tol = difference between two consecutive chi-square values considered small
#           if the difference is less than this value for two iterations in a row
#           the fit is considered to have converged.
def lev_mar(f,g, x, y, err, guess, max_iter=100, rel_tol=1E-3):
    # Produce initial guess data, and print output
    p = guess
    model = f(x,p)
    chisq = chisqr(y, model, err)

    print("Initial guess: %s" % p)
    print("Gives 𝛘² = %.3f\n" % chisqr(y, model, err))
    
    # Initialize end conditions and L-M lambda factor
    lmfac = 0.001
    done = False
    small = False

    for i in range(max_iter):
        # Check end condition
        if done:
            break

        # Perform next step
        increase = True
        # If lambda was increased
        while increase:
            print("Lambda set to: %e" % lmfac)

            # Usual L-M Step
            grad = np.matrix(g(x,p))
            r = np.matrix(resid(y, model)).transpose()
            # This is where the magic happens
            lhs = grad.transpose() * np.diag(1.0/err**2) * grad + lmfac * np.eye(grad.shape[1])
            rhs = grad.transpose() * np.diag(1.0/err**2) * r
            dp  = np.linalg.inv(lhs) * rhs
            dp  = np.squeeze(np.asarray(dp)) # Squeeze as array to get back 1D array

            # Levenberg-Marquard lambda factor update
            t_model = f(x, p + dp) # Test Model with new parameters
            t_chisq = chisqr(y, t_model, err)
            # If chisq got worse, increase lambda, else decrease it
            if t_chisq >= chisq:
                lmfac *= 10
            else:
                lmfac *= 0.1
                increase = False
                # If the decrease was really small flag for end
                if chisq - t_chisq < rel_tol:
                    # If decrease was small twice, consider converged
                    if small:
                        done = True
                        print("Converged!")
                    # Otherwise, indicate that it was small once
                    else:
                        small = True
                # Otherwise, reset small flag
                else:
                    small = False
        
        # Update parameters and model
        p += dp
        model = t_model
        chisq = t_chisq

        print("Δparams: %s" % dp)
        print("Parameters: %s" % p)
        print("Gives 𝛘² = %.3f\n" % chisq)


    return p, chisq


# Define functions of fixed tau and guess parameters
func = lambda x, p: get_spectrum(x, np.insert(p,3,0.05))
ds = list(zip([0,1,2,4,5], np.array([65,0.02,0.1,2e-9,0.96])/70))
grad = lambda x, p: get_grad(x, np.insert(p,3,0.05), ds)
guess = np.asarray([65,0.02,0.1,2e-9,0.96])

# Do the fitting
p, chisq = lev_mar(func, grad, ls, wmap[:,1], wmap[:,2], guess)
# Plot Results
fig, ax = plt.subplots(2,1, sharex=True)
res_ax = ax[0]
dat_ax = ax[1]

# Reinsert tau into parameters
p = np.insert(p,3,0.05)
model = get_spectrum(ls, p)
print(chisqr(wmap[:,1], model, wmap[:,2]))
stud_res = studentized(wmap[:,1], model, wmap[:,2])
plot_res(res_ax, wmap[:,0], stud_res)
plot_dat(dat_ax, wmap[:,0], wmap[:,1], wmap[:,2])
plot_func(dat_ax, ls, model, label="fit")
dat_ax.legend()
res_ax.set_ylabel("Studentized Residual")
dat_ax.set_ylabel("CMB PS")
dat_ax.set_xlabel("l")
plt.show(block = True)

# Results:
# Compute covarient matrix from final results
grad_no_tau = np.matrix(get_grad(ls,p,ds))
cov_no_tau = np.linalg.inv(grad_no_tau.transpose() * np.diag(1.0/wmap[:,2]**2) * grad_no_tau)
print("~~~LM Fit Results~~~")
print("Output Parameters:")
print(p)
print("Fit Errors No Tau:")
print(np.sqrt(np.diag(cov_no_tau)))

# Do the same, but as if we had fit for tau. We use this one for MCMC
full_ds = list(zip([0,1,2,3,4,5], p/70))
grad = np.matrix(get_grad(ls,p,full_ds))
cov = np.linalg.inv(grad.transpose() * np.diag(1.0/wmap[:,2]**2) * grad)
print("Fit Errors With Tau:")
print(np.sqrt(np.diag(cov)))
########
# MCMC #
########
# Generate a parameter step with a given covariance matrix.
def cov_step(covmat):
    chol = np.linalg.cholesky(covmat)
    step = np.squeeze(np.asarray(chol @ np.random.randn(chol.shape[0])))
    return step

# MCMC routine sort of hardcoded for this problem.
def monte(params, nstep):
    # Initiate output vectors
    chains = np.zeros([nstep, len(params)])
    chisqvec = np.zeros(nstep)

    # Initial model and chisqr
    cmb = get_spectrum(ls, params)
    chisq = chisqr(wmap[:,1], cmb, wmap[:,2])
    
    # Holds number of accepted steps for calculating percent.
    accept = 0
    # Scale of each mcmc step, hand tweaked...
    scale = 0.4

    print("Starting MCMC")
    
    # Pathname for output file
    path = "chain" + datetime.now().strftime("%d-%H-%M-%S")
    # Write header
    with open(path,"w+") as f:
        f.write("H0,ombh2,omch2,tau,As,ns,chisq,accept\n")
    # Write initial values
    write(path, params, chisq, 100.0)
    
    # Doing the MCMC
    for i in range(nstep):

        # Generate a new step, ensuring tau is greater than zero
        n_params = params + cov_step(cov) * scale
        while n_params[3] <= 0:
            n_params = params + cov_step(cov) * scale
        
        # Calculating the new chi-squared value
        n_chisq = chisqr(wmap[:,1],get_spectrum(ls, n_params),wmap[:,2])
        # Checking the delta, and wether we take the step or not
        delta = n_chisq - chisq
        prob = np.exp(-0.5 * delta)
        if np.random.rand(1) < prob:
            # Step taken, update parameters
            params = n_params
            chisq = n_chisq
            accept += 1
        # Print current parameters
        for param in params:
            print("%.2e, " % param, end="")
        print("%.2f " % chisq, end="")
        print("%.2f %%" % (accept/(i+1) * 100))

        # Add to output values
        chains[i,:] = params
        chisqvec[i] = chisq
        # Write results to file
        write(path, params, chisq, accept/(i+1) * 100)
        # Print iteration number
        print(i)
    return chains

chains = monte(params, steps)
params = np.mean(chains,axis=0)
errors = np.std(chains, axis=0)
cmb = get_spectrum(ls, params)
chisq = chisqr(wmap[:,1], cmb, wmap[:,2])
fig, ax = plt.subplots(2,1, sharex=True)
res_ax = ax[0]
dat_ax = ax[1]

plot_res(res_ax, wmap[:,0], studentized(wmap[:,1], cmb, wmap[:,2]))
plot_dat(dat_ax, wmap[:,0], wmap[:,1], wmap[:,2])
plot_func(dat_ax, ls, cmb, label="fit")
dat_ax.legend()

print("𝛘² = %.3f" % chisq)
plt.show(block=True)

plt.plot(chains[:,0])
plt.plot(chains[:,1])
plt.plot(chains[:,2])
plt.plot(chains[:,3])
plt.plot(chains[:,4])
plt.plot(chains[:,5])
plt.show(block=True)
