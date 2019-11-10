import numpy as np
import scipy.ndimage as ndimage
import scipy.signal as sig
import matplotlib.mlab as mlab
from matplotlib import pyplot as plt
import h5py
import glob
plt.style.use("ggplot")

dataFolder = "./data"
fs = 4096.0

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

# Initialize arrays to holsignal.welchd spectra averages
# One for each detector
h_spect = []
l_spect = []

# Loop over datasets, adding to average
# spectra of the detector it belongs to
for fname in datasets:
    # Read Data
    strain,dt,utc=read_file(fname)

    # Take full psd of each signal
    # Defaults to hanning window
    spectr, freq = mlab.psd(strain,Fs=fs,NFFT=len(strain))
    # Average with proper detector
    if "H-H1" in fname:
        h_spect.append(spectr)
    elif "L-L1" in fname:
        l_spect.append(spectr)

h_spect = np.mean(np.vstack(h_spect),axis=0)
l_spect = np.mean(np.vstack(l_spect),axis=0)

# Average together some neighbouring points to smooth a bit
# Generate a plot of various smoothing to gauge what's good
for n in [1,2,3,4,5]:
    plt.loglog(np.sqrt(moving_average(h_spect,n)))
plt.show(block=True)

# 3 looks pretty good as it avoids flat tops of peaks but also
# removes jagged spikes and a bit of noise.
h_spect = moving_average(h_spect,3)
l_spect = moving_average(l_spect,3)

# Plot the noise model
plt.figure()
plt.loglog(freq, np.sqrt(h_spect), label="Hanford")
plt.loglog(freq, np.sqrt(l_spect), label="Livingston")
plt.legend()
plt.axis([20, 2000, 1e-24, 1e-19])
plt.show(block=True)

plt.figure()

def whiten(strain,noise):
    # Window the given data
    strain = strain * sig.blackman(len(strain))
    spectr = np.fft.rfft(strain)
    # To whiten each data set, we divide by our noise model in fourier space
    spectr = spectr / np.sqrt(noise)
    # From example code: something about normalization
    spectr = spectr * 1.0 / np.sqrt(fs/2)
    white_strain = np.fft.irfft(spectr, n=len(strain))

    # apply bandpass from 20 to 2000
    bb,ba = sig.butter(5, np.array([20,2000]) / (fs/2), btype="band")
    return sig.lfilter(bb,ba,white_strain)


# Loop through each dataset to find events
for fname in datasets:

    print('reading file ',fname)
    strain,dt,utc=read_file(fname)

    if "H-H1" in fname:
        noise = h_spect
    elif "L-L1" in fname:
        noise = l_spect

    white_strain = whiten(strain,noise)

    for template_name in templates:
        th,tl=read_template(template_name)
        print('\tTrying template ', template_name)
        if "H-H1" in fname:
            template = th
        elif "L-L1" in fname:
            template = tl

        white_temp = whiten(template, noise)

        template_spectr = np.fft.rfft(template) / fs
        strain_spectr = np.fft.rfft(white_strain) / fs

        opt = strain_spectr * template_spectr.conjugate() / noise
        opt_time = 2 * np.fft.irfft(opt) * fs

        plt.plot(np.abs(opt_time))
        plt.show(block=True)

"""
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
