import numpy as np
import scipy.ndimage as ndimage
import scipy.signal as signal
from matplotlib import pyplot as plt
import h5py
import glob
plt.style.use("ggplot")

dataFolder = "./data"
fs = 4096

def read_template(filename):
    dataFile=h5py.File(filename,'r')
    template=dataFile['template']
    th=template[0]
    tl=template[1]
    return th,tl

def read_file(filename):
    dataFile=h5py.File(filename,'r')
    #dqInfo = dataFile['quality']['simple']
    #qmask=dqInfo['DQmask'][...]

    meta=dataFile['meta']
    #gpsStart=meta['GPSstart'].value

    utc=meta['UTCstart'][()]
    duration=meta['Duration'][()]
    strain=dataFile['strain']['Strain'][()]
    dt=(1.0*duration)/len(strain)

    dataFile.close()
    return strain,dt,utc

def moving_average(a, n=10):
    return ndimage.uniform_filter(a,n)

templates = glob.glob(dataFolder + "/*template*")
datasets  = glob.glob(dataFolder + "/*LOSC*.hdf5")

###############
# Noise Model #
###############

# Initialize arrays to hold spectra averages
# One for each detector
h_spect = np.zeros(len(read_file(datasets[0])[0])//16 + 1)
l_spect = np.zeros(len(read_file(datasets[0])[0])//16 + 1)

# Loop over datasets, adding to average 
# spectra of the detector it belongs to
for fname in datasets:
    # Read Data
    strain,dt,utc=read_file(fname)

    # Window with blackman function
    #strain = strain * np.blackman(len(strain))
    #spectr = np.abs(np.fft.rfft(strain))**2

    # Take psd of each trace with welch's method
    # Defaults to hanning window
    spectr, freq = signal.welch(strain, fs=fs, nperseg=4*fs, detrend=0)

    # Average with proper detector
    if "H-H1" in fname:
        h_spect = (h_spect + spectr)/2
    elif "L-L1" in fname:
        l_spect = (h_spect + spectr)/2

# Generate the frequencies that go along with the last loaded dataset
# Assuming they're all the same
freq = np.fft.rfftfreq(len(strain),dt)

# Average together some neighbouring points to smooth a bit
# Generate a plot of various smoothing to gauge what's good
for n in [1,5,10,20]:
    plt.loglog(np.sqrt(moving_average(h_spect,n)))
plt.show(block=True)

# 5 looks pretty good as it avoids flat tops of peaks but also
# removes jagged spikes and a bit of noise.
h_spect = moving_average(h_spect,5)
l_spect = moving_average(l_spect,5)

# Plot the noise model
plt.figure()
plt.loglog(freq, np.sqrt(h_spect), label="Hanford")
plt.loglog(freq, np.sqrt(l_spect), label="Livingston")
plt.legend()
plt.axis([20, 2000, 1e-24, 1e-19])
plt.show(block=False)

"""
# Loop through each dataset to find events
for fname in datasets:

    print('reading file ',fname)
    strain,dt,utc=read_file(fname)

    for template_name in templates:
        th,tl=read_template(template_name)


#spec,nu=measure_ps(strain,do_win=True,dt=dt,osamp=16)
#strain_white=noise_filter(strain,np.sqrt(spec),nu,nu_max=1600.,taper=5000)

#th_white=noise_filter(th,np.sqrt(spec),nu,nu_max=1600.,taper=5000)
#tl_white=noise_filter(tl,np.sqrt(spec),nu,nu_max=1600.,taper=5000)


#matched_filt_h=np.fft.irfft(np.fft.rfft(strain_white)*np.conj(np.fft.rfft(th_white)))
#matched_filt_l=np.fft.irfft(np.fft.rfft(strain_white)*np.conj(np.fft.rfft(tl_white)))




#copied from bash from class
# strain2=np.append(strain,np.flipud(strain[1:-1]))
# tobs=len(strain)*dt
# k_true=np.arange(len(myft))*dnu
"""
