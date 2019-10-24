import numpy as np
from sys import argv
from matplotlib import pyplot as plt
from itertools import combinations
plt.style.use("ggplot")

data = np.genfromtxt(argv[1], skip_header=1, delimiter=',')

chisq = data[:,6]
chains = data[:,:6]
for i in range(chains.shape[1]):
    plt.plot(chains[:,i])
    plt.show(block=True)

print(chains)
fft = np.abs(np.fft.rfft(chains, axis=0))
print(fft.shape,chains.shape)
chainfig, c_ax = plt.subplots(2,1)
for i in range(chains.shape[1]):
    c_ax[0].plot(chains[:,i]/max(chains[:,i]))
    c_ax[1].semilogy(fft[:,i]/max(fft[:,i]))
plt.show(block=True)

corr, ax = plt.subplots(5,5)
for pair in combinations(enumerate(chains.transpose()), 2):
    print(pair)
    axis = ax[pair[0][0], pair[1][0]-1]
    axis.scatter(pair[0][1]/max(pair[0][1]), pair[1][1]/max(pair[1][1]))
plt.show(block=True)

