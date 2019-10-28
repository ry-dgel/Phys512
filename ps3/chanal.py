import numpy as np
from sys import argv
from matplotlib import pyplot as plt
plt.style.use("ggplot")

# Load in specified chain file
data = np.genfromtxt(argv[1], skip_header=1, delimiter=',')
# Extract chi-square data
chisq = data[:,6]
# Extract accepted percentage
accepted = data[:,7]
# Extract parameter chains
chains = data[:,:6]

# Plot the chisquare as a function of time, converged area can roughly
# be taken as when the chi-square value flattens out.
plt.plot(chisq)
plt.show(block=True)
# Plot the accepted parameter percentage as a function of time,
# converged are can roughly be taken as when this value flattens out to
# hopefully ~25%
plt.plot(accepted)
plt.show(block=True)

# Get cutoff value from input
cutoff = input("Cutoff start? ")
cutoff = int(cutoff)
# Truncate chains from cutoff till end
chains = chains[cutoff:,:]

def analyze_chain(chains):
    # Print out the parameters and errors by taking the mean and std
    # over the chains
    print("Parameter means are:")
    print(np.mean(chains, axis=0))
    print("Parameter errors are:")
    print(np.std(chains, axis=0))

    # Inititate plot of chains
    chainfig, c_ax = plt.subplots(2,1)
    for i in range(chains.shape[1]):
        # Plot the parameter chain, normalized for each parameter
        c_ax[0].plot(chains[:,i]/chains[:,i].max(), label=str(i))
    c_ax[0].legend()

    for i in range(chains.shape[1]):
        # Take the real value fourier transform of the chain
        cft = np.fft.rfft(chains[:,i])
        # Generate x value points
        x = np.arange(cft.size)
        # Divide by half to only plot positive values
        x = x[1:cft.size//2]
        cft = np.abs(cft[1:cft.size//2])
        # Plot the fourier transform of each chain, normalized.
        c_ax[1].loglog(x, cft/cft.max())
    plt.show(block=True)

def analyze_chain_prior(chains):
    tau_true = 0.0544
    tau_error = 0.0073
    weight = np.exp(-0.5 * ((chains[:,3] - tau_true)/tau_error)**2)
    normalize = np.sum(weight)
    # Print out the parameters and errors by taking the mean and std
    # over the chains
    print("Importance Sampled Parameters are:")
    p = [np.sum(chains[:,i] * weight)/normalize for i in range(chains.shape[1])]
    print(p)
    # Inititate plot of chains
    chainfig, c_ax = plt.subplots(2,1)
    for i in range(chains.shape[1]):
        # Plot the parameter chain, normalized for each parameter
        c_ax[0].plot(chains[:,i]/chains[:,i].max(), label=str(i))
    c_ax[0].legend()

    for i in range(chains.shape[1]):
        # Take the real value fourier transform of the chain
        cft = np.fft.rfft(chains[:,i])
        # Generate x value points
        x = np.arange(cft.size)
        # Divide by half to only plot positive values
        x = x[1:cft.size//2]
        cft = np.abs(cft[1:cft.size//2])
        # Plot the fourier transform of each chain, normalized.
        c_ax[1].loglog(x, cft/cft.max())
    plt.show(block=True)

analyze_chain(chains)
analyze_chain_prior(chains)


