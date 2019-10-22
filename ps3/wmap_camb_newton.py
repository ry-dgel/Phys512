import numpy as np
import camb
from matplotlib import pyplot as plt
plt.style.use("ggplot")
def get_spectrum(pars,lmax=2000):
    H0=pars[0]
    ombh2=pars[1]
    omch2=pars[2]
    tau=pars[3]
    As=pars[4]
    ns=pars[5]

    pars=camb.CAMBparams()
    pars.set_cosmology(H0=H0,ombh2=ombh2,omch2=omch2,mnu=0.06,omk=0,tau=tau)
    pars.InitPower.set_params(As=As,ns=ns,r=0)
    pars.set_for_lmax(lmax,lens_potential_accuracy=0)

    results=camb.get_results(pars)
    powers=results.get_cmb_power_spectra(pars,CMB_unit='muK')
    cmb=powers['total']
    tt=cmb[:,0]

    return tt

def get_grad(pars, ds, xs, length):
    grad = np.zeros((len(pars), length))
    
    for pair in ds:
        i = pair[0]
        d = pair[1]

        delta = np.zeros(len(pars))
        delta[i] = d

        plus = get_spectrum(pars + delta)[xs.astype(int)]
        mins = get_spectrum(pars - delta)[xs.astype(int)]
        grad[i,:] = (plus - mins)/(2 * d)

    return grad

def plot_res(ax, x, stud_res):
    ax.errorbar(x, stud_res, 1, fmt='o', markersize=3)

def plot_dat(ax, x, y, e, f):
    ax.errorbar(x,y,e, fmt='o', markersize=3)
    ax.plot(x,f, zorder=1000)

pars=np.asarray([65,0.02,0.1,0.05,2e-9,0.96])
wmap=np.loadtxt('wmap_tt_spectrum_9yr_v5.txt')

# Truncate cmb data to ls of interest
cmb = get_spectrum(pars)
cmb = cmb[wmap[:,0].astype(int)]

ds = [(0, 0.001),(1, 0.001),(2,0.001),(4,1E-11),(5,0.001)]
grad = get_grad(pars, ds, wmap[:,0], len(cmb))

stud_res = (wmap[:,1] - cmb)/wmap[:,2]
chi_sqr  = np.sum(stud_res**2)

fig, ax = plt.subplots(2,1, sharex=True)
res_ax = ax[0]
dat_ax = ax[1]

plot_res(res_ax, wmap[:,0], stud_res)
plot_dat(dat_ax, wmap[:,0], wmap[:,1], wmap[:,2], cmb)

print("𝛘² = %.3f" % chi_sqr)
plt.show(block=True)
