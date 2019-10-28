import camb
import numpy as np
import os
import psutil
process = psutil.Process(os.getpid())

params=np.asarray([65,0.02,0.1,0.05,2e-9,0.96])
ls = range(2,2000)

H0=params[0]
ombh2=params[1]
omch2=params[2]
tau=params[3]
As=params[4]
ns=params[5]

previous = process.memory_info().rss
print(previous)
while True:
    pars=camb.CAMBparams()
    pars.set_cosmology(H0=H0,ombh2=ombh2,omch2=omch2,mnu=0.06,omk=0,tau=tau)
    pars.InitPower.set_params(As=As,ns=ns,r=0)
    pars.set_for_lmax(int(max(ls)),lens_potential_accuracy=0)

    results = camb.get_results(pars)
    del results
    new = process.memory_info().rss 
    print("Current Memory Usage: %d. " % (new), end="")
    print("Loop Delta: %d" % (new - previous))
    previous = new
