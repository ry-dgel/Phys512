import numpy as np
import camb
from matplotlib import pyplot as plt
plt.style.use("ggplot")

##########
# Params #
##########
nstep = 10000
params=np.asarray([65,0.02,0.1,0.05,2e-9,0.96])
scale_factor = params/100

########
# Data #
########
wmap=np.loadtxt('wmap_tt_spectrum_9yr_v5.txt')

#############
# Functions #
#############
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

def dum_step(params):
    return np.random.rand(len(params))

def cov_step(covmat):
    chol = np.linalg.cholesky(covmat)
    return chol @ np.random.rand(covmat.shape[0])

def plot_res(ax, x, stud_res):
    ax.errorbar(x, stud_res, 1, fmt='o', markersize=3)

def plot_dat(ax, x, y, e, f):
    ax.errorbar(x,y,e, fmt='o', markersize=3)

def plot_func(ax, x, f):
    ax.plot(x,f, zorder=1000)

def studentized(wmap, cmb):
    return (wmap[:,1] - cmb)/wmap[:,2]

# Truncate cmb data to ls of interest
ls = wmap[:,0]
cmb = get_spectrum(ls, params)

chains = np.zeros([nstep, len(params)])
chisqvec = np.zeros(nstep)
stud_res = studentized(wmap, cmb)
chisq = np.sum(stud_res**2)

init_params = params
init_guess = cmb

for i in range(nstep):
    n_params = params + dum_step(params) * scale_factor
    while n_params[3] <= 0:
        n_params = params + dum_step(params) * scale_factor
    n_model = get_spectrum(ls, n_params)

    n_chisq = np.sum(studentized(wmap,n_model))

    delta = n_chisq - chisq
    prob = np.exp(-0.5 * delta)
    if np.random.rand(1) < prob:
        params = n_params
        cmb = n_model
        chisq = n_chisq
    chains[i,:] = params
    chisqvec[i] = chisq
    print(i,end="\r")

fig, ax = plt.subplots(2,1, sharex=True)
res_ax = ax[0]
dat_ax = ax[1]

plot_res(res_ax, wmap[:,0], stud_res)
plot_dat(dat_ax, wmap[:,0], wmap[:,1], wmap[:,2])
plot_func(dat_ax, ls, init_guess)
plot_func(dat_ax, ls, cmb)

print("𝛘² = %.3f" % chisq)
plt.show(block=True)

plt.plot(chains[:,0])
plt.plot(chains[:,1])
plt.plot(chains[:,2])
plt.plot(chains[:,3])
plt.plot(chains[:,4])
plt.plot(chains[:,5])
plt.show(block=True)
