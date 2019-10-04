import numpy as np
import matplotlib.pyplot as plt

delta_chi = 1
max_iters = 50

def exp_decay(t, p, t0):
    #p[0] = offset
    #p[1] = amplitude
    #p[2] = decay rate
    y = p[0] + p[1] * np.exp(-(t-t0)/p[2])

    grad=np.zeros([t.size,p.size])
    grad[:,0] = np.ones(len(t))
    grad[:,1] = np.exp(-(t-t0)/p[2])
    grad[:,2] = p[1] * (t-t0) / (p[2]**2) * np.exp(-(t-t0)/p[2])

    return y,grad

data = np.genfromtxt('229614158_PDCSAP_SC6.txt', delimiter=',')
time = data[:,0]
flux = data[:,1]
t0 = time[np.argmax(flux)]
error = np.std(flux[time < t0])

flux = flux[time >= t0]
time = time[time >= t0]
flux = flux[time <= 1707.2]
time = time[time <= 1707.2]



func = lambda t, p: exp_decay(t,p,t0)
guess = np.array([1.0, 0.26, 0.05])
init, _ = func(time, guess)

p = guess.copy()

old_chi2 = np.inf

for i in range(max_iters):
    pred, grad = func(time, p)
    res = flux - pred
    chi2 = np.square(res/error).sum()
    print("With parameters: %s," % p)
    print("Iteration %d gives chi2 of %f\n" % (i, chi2))

    res = np.matrix(res).transpose()
    grad = np.matrix(grad)

    lhs = grad.transpose() * grad
    rhs = grad.transpose() * res
    dp  = np.linalg.inv(lhs) * rhs
    p = p + np.squeeze(np.asarray(dp))

    if old_chi2 - chi2 < delta_chi:
        print("Converged!")
        break
    if i == max_iters - 1:
        print("Didn't Converge!")
    old_chi2 = chi2

plt.figure(dpi=255)
plt.scatter(time,flux, label="data", marker="x")
plt.errorbar(time,flux,yerr=error, linestyle="None")
plt.plot(time, pred, color="red", label="fit")
plt.plot(time, init, color="gray", label="guess")
plt.xlabel("Time")
plt.ylabel("Flux")
plt.title("Flare Fit")
plt.legend()
plt.show(block=True)
