import numpy as np
import matplotlib.pyplot as plt

def fun0(x):
    return np.exp(x)

def fun1(x):
    return np.exp(0.01 * x)

def truder0(x):
    return np.exp(x)

def truder1(x):
    return 0.01 * np.exp(0.01 * x)

def fifthder0(x):
    return np.exp(x)

def fifthder1(x):
    return (0.01)**5 * np.exp(0.01 * x)

def deriv(fun,x,dx):
    return (fun(x+dx) - fun(x-dx) - (fun(x + 2*dx) - fun(x - 2*dx))/8) * 2 / (3*dx)

x = 1
dx = np.linspace(1,1E-16,10000)

fig,ax = plt.subplots(1,1)
ax.loglog(dx, np.abs(deriv(fun0,x,dx) - truder0(x)), label="exp(x) derivative error")
ax.axvline(((30/4) * fun0(x) * np.finfo(float).eps / fifthder0(x))**(1/5),label="expected exp(x) optimal", linestyle="--")
ax.loglog(dx, np.abs(deriv(fun1,x,dx) - truder1(x)),color="orange", label="exp(0.01x) derivative error")
ax.axvline(((30/4) * fun1(x) * np.finfo(float).eps / fifthder1(x))**(1/5), label="expected exp(0.01x) optimal",color="orange", linestyle="--")
ax.set_xlabel("dx")
ax.set_ylabel("Absolute Error")
ax.legend()
plt.show(block=True)
