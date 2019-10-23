import numpy as np
import camb
from matplotlib import pyplot as plt
from datetime import datetime
import sys
plt.style.use("ggplot")

##########
# Params #
##########
nstep = 10000
params=np.asarray([65,0.02,0.1,0.05,2e-9,0.96])
ds = list(zip([0,1,2,4,5], np.array([65,0.02,0.1,2e-9,0.96])/200))

########
# Data #
########
wmap=np.loadtxt('wmap_tt_spectrum_9yr_v5.txt')
ls = wmap[:,-1]

#############
# Functions #
#############
def write(f, params, chisq):
    for param in params:
        f.write("%e," % param)
    f.write("%f\n" % chisq)
    f.flush()

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

    results=camb.get_results(pars)
    powers=results.get_cmb_power_spectra(pars,CMB_unit='muK')
    cmb=powers['total']
    tt=cmb[:,0]

    return tt[ls.astype(int)]

def get_grad(ls, pars, ds):
    grad = np.zeros((len(ls),len(ds)))

    for i,pair in enumerate(ds):
        idx = pair[0]
        d = pair[1]

        delta = np.zeros(len(pars))
        delta[idx] = d

        plus = get_spectrum(ls, pars + delta)
        mins = get_spectrum(ls, pars - delta)
        grad[:,i] = (plus - mins)/(2 * d)

    return grad

def resid(true, fit):
    return true - fit

def studentized(true, fit, error):
    return resid(true,fit)/error

def chisqr(true,fit,error):
    return np.sum(np.square(studentized(true,fit,error)))

#######################
# Levenberg-Marquardt #
#######################
def lev_mar(f,g, x, y, err, guess, max_iter=100, rel_tol=1E-3):
    p = guess
    model = f(x,p)
    chisq = chisqr(y, model, err)

    print("Initial guess: %s" % p)
    print("Gives 𝛘² = %.3f" % chisqr(y, model, err))

    lmfac = 0.001
    done = False
    small = False

    for i in range(max_iter):
        # Check end condition
        if done:
            break

        # Perform next step
        increase = True
        while increase:
            print("Lambda set to: %.3f" % lmfac)
            grad = np.matrix(g(x,p))
            r = np.matrix(resid(y, model)).transpose()

            lhs = grad.transpose() * grad + lmfac * np.eye(grad.shape[1])
            rhs = grad.transpose() * r
            dp  = np.linalg.inv(lhs) * rhs
            dp  = np.squeeze(np.asarray(dp))

            # Levenberg-Marquard lambda factor update
            t_model = f(x, p + dp)
            t_chisq = chisqr(y, t_model, err)
            if t_chisq >= chisq:
                lmfac *= 10
            else:
                lmfac *= 0.1
                increase = False
                if chisq - t_chisq < rel_tol:
                    if small:
                        done = True
                    else:
                        small = True

        p += dp
        model = t_model
        chisq = t_chisq
        print("Parameters: %s" % p)
        print("Gives 𝛘² = %.3f" % chisq)

    grad = np.matrix(g(x,p))
    cov = grad.transpose() * grad

    return p, chisq, cov

func = lambda x, p: get_spectrum(x, np.insert(p,3,0.05))
grad = lambda x, p: get_grad(x, np.insert(p,3,0.05), ds)
guess = np.asarray([65,0.02,0.1,2e-9,0.96])
p, chisq = lev_mar(func, grad, wmap[:,0], wmap[:,1], wmap[:,2], guess)

lmfac = 0
cov = np.cov()
########
# MCMC #
########

def cov_step(covmat):
    chol = np.linalg.cholesky(covmat)
    return chol @ np.random.randn(covmat.shape[0])

############
# Plotting #
############

def plot_res(ax, x, stud_res):
    ax.errorbar(x, stud_res, 1, fmt='o', markersize=3)

def plot_dat(ax, x, y, e):
    ax.errorbar(x,y,e, fmt='o', markersize=3)

def plot_func(ax, x, f, label=""):
    ax.plot(x,f, zorder=1000, label=label)

"""
cmb = get_spectrum(ls, params)
chains = np.zeros([nstep, len(params)])
chisqvec = np.zeros(nstep)
stud_res = studentized(wmap, cmb)
chisq = chisqr(wmap,cmb)

init_params = params
init_guess = cmb

with open("chain" + datetime.now().strftime("%d-%H-%M-%S"),"w") as f:
    f.write("H0,ombh2,omch2,tau,As,ns,chisq\n")
    write(f, params, chisq)
    for i in range(nstep):
        n_params = params + cov_step(cov)
        while n_params[3] <= 0:
            n_params = params + cov_step(cov)
        n_cmb = get_spectrum(ls, n_params)

        stud_res = studentized(wmap, n_cmb)
        n_chisq = chisqr(wmap,n_cmb)

        delta = n_chisq - chisq
        prob = np.exp(-0.5 * delta)
        if np.random.rand(1) < prob:
            params = n_params
            cmb = n_cmb
            chisq = n_chisq
        chains[i,:] = params
        chisqvec[i] = chisq

        write(f, params, chisq)
        print(i,end="\r")

fig, ax = plt.subplots(2,1, sharex=True)
res_ax = ax[0]
dat_ax = ax[1]

plot_res(res_ax, wmap[:,0], stud_res)
plot_dat(dat_ax, wmap[:,0], wmap[:,1], wmap[:,2])
plot_func(dat_ax, ls, init_guess, label="Guess")
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
"""
