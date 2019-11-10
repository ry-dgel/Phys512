import numpy as np
import scipy.ndimage as ndimage
import scipy.signal as sig
import scipy.interpolate as intp
from matplotlib import pyplot as plt
import h5py
import glob
import json

plt.style.use("ggplot")

dataFolder = "./data"

# Why was this labelled as h and l? The two data sets are for
# different polarization...
def read_template(filename):
    dataFile=h5py.File(filename,'r')
    template=dataFile['template']
    th=template[0]
    tl=template[1]
    return th,tl

def read_file(filename):
    dataFile=h5py.File(filename,'r')
    meta=dataFile['meta']

    utc=meta['UTCstart'][()]
    duration=meta['Duration'][()]
    strain=dataFile['strain']['Strain'][()]
    dt=(1.0*duration)/len(strain)

    dataFile.close()
    return strain,dt,utc

def moving_average(a, n=5):
    return ndimage.uniform_filter(a,n)

def psd(signal, dt):
    # Window Signal
    window = sig.blackman(len(signal))
    signal = signal * window
    # Take rfft
    psd = np.fft.rfft(signal)
    # Calculate psd and normalize accordingly
    psd = 2 * dt* np.abs(psd)**2 / np.sum(window**2)
    # Generate frequencies of output psd
    freqs = np.fft.rfftfreq(len(signal), dt)
    return psd, freqs


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
    spectr, freq = psd(strain,dt)
    # Average with proper detector
    if "H-H1" in fname:
        h_spect.append(spectr)
    elif "L-L1" in fname:
        l_spect.append(spectr)

h_spect = np.mean(np.vstack(h_spect),axis=0)
l_spect = np.mean(np.vstack(l_spect),axis=0)

# Average together some neighbouring points to smooth a bit
# Generate a plot of various smoothing to gauge what's good
plt.figure()
for n in [1,3,5,10,15,20,30]:
    plt.loglog(np.sqrt(moving_average(h_spect,n)))
plt.show(block=True)

# 5 looks pretty good as it avoids flat tops of peaks but also
# removes jagged spikes and a bit of noise.
h_spect = moving_average(h_spect,5)
l_spect = moving_average(l_spect,5)
h_itp = intp.interp1d(freq, h_spect)
l_itp = intp.interp1d(freq, l_spect)

# Plot the noise model
plt.figure()
plt.loglog(freq, np.sqrt(h_spect), label="Hanford")
plt.loglog(freq, np.sqrt(l_spect), label="Livingston")
plt.legend()
plt.axis([20, 2000, 1e-24, 1e-19])
plt.show(block=True)

##################
# Matched Filter #
##################

def whiten(strain,noise,dt):
    spectr = np.fft.fft(strain)
    freq = np.fft.fftfreq(len(strain),dt)
    # To whiten each data set, we divide by our noise model in fourier space
    spectr = spectr / np.sqrt(noise(np.abs(freq)))
    # From example code: something about normalization
    spectr = spectr * 1.0 / np.sqrt(1.0/(dt * 2))
    white_strain = np.fft.ifft(spectr, n=len(strain))

    # apply bandpass from 20 to 2000
    bb,ba = sig.butter(5, np.array([20,2000]) * 2 * dt, btype="band")
    norm = np.sqrt((2000-20)*2*dt)
    return sig.lfilter(bb,ba,white_strain) / norm

def match_filter(strain, template, noise, dt):
    window = sig.blackman(strain.size)

    # Take the fft of the data and template
    strain_spect = np.fft.fft(strain*window) * dt
    temp_spect   = np.fft.fft(template*window) * dt
    freq = np.fft.fftfreq(len(window), dt)
    df = np.average(np.diff(freq))
    noise_itp = noise(np.abs(freq))

    # Do the matched filter
    optimal = strain_spect * temp_spect.conjugate() / noise_itp

    # Inverse fourier transform and scale to go back to time.
    optimal_time = 2 * np.fft.ifft(optimal) / dt

    # Calculate sigma for getting singal to noise.
    sigmasq = np.sum(temp_spect * temp_spect.conjugate() / noise_itp) * df
    sigma = np.sqrt(np.abs(sigmasq))
    SNR = optimal_time / sigma
    print("Sigma = ", sigma)

    return SNR


# Loop through each dataset to find events
with open("data/BBH_events_v3.json") as json_file:
    json_data = json.load(json_file)

for event in json_data.keys():
    event_json = json_data[event]
    print('reading event', event_json['name'])

    strain_h,dt_h,utc_h=read_file(dataFolder + "/" + event_json['fn_H1'])
    strain_l,dt_l,utc_l=read_file(dataFolder + "/" + event_json['fn_L1'])

    wstrain_h = whiten(strain_h, h_itp, dt_h)
    wstrain_l = whiten(strain_l, l_itp, dt_l)

    # Lets keep things interesting and pretend we don't know which
    # template we should use.
    fig,ax = plt.subplots(len(templates),2)
    for idx, template_name in enumerate(templates):
        t_p,t_c=read_template(template_name)
        print('\tTrying template ', template_name)
        temp = (t_p + t_c * 1.0j)

        SNR_h = (match_filter(wstrain_h, temp, h_itp, dt_h))
        SNR_l = (match_filter(wstrain_l, temp, l_itp, dt_l))

        idxmax_h = np.argmax(np.abs(SNR_h))
        idxmax_l = np.argmax(np.abs(SNR_l))
        print(np.angle(SNR_h[idxmax_h]))
        wtemp_h = np.real(whiten(temp * SNR_h[idxmax_h] / abs(SNR_h[idxmax_h]), h_itp, dt_h))
        wtemp_l = np.real(whiten(temp * SNR_h[idxmax_l] / abs(SNR_h[idxmax_l]), l_itp, dt_h))

        ax[idx,0].plot(np.abs(SNR_h), label="H")
        ax[idx,0].plot(np.abs(SNR_l), label="L")
        ax[idx,0].set_ylabel(template_name.split("_")[0].split("/")[2])


        offset = int(round(0.1/dt))

        gwave_h = wstrain_h[idxmax_h-offset:idxmax_h+offset]
        twave_h = wtemp_h[len(wtemp_h)//2 - offset:len(wtemp_h)//2 + offset]
        ax[idx,1].plot(np.linspace(-0.1,0.1,len(gwave_h)), gwave_h/max(gwave_h)+1.0)

        gwave_l = wstrain_l[idxmax_l-offset:idxmax_l+offset]
        twave_l = wtemp_l[len(wtemp_l)//2 - offset:len(wtemp_l)//2 + offset]
        ax[idx,1].plot(np.linspace(-0.1,0.1,len(gwave_l)), gwave_l/max(gwave_l)-1.0)

        ax[idx,1].plot(np.linspace(-0.1,0.1,len(gwave_h)), twave_h/max(twave_h)+1.0)
        ax[idx,1].plot(np.linspace(-0.1,0.1,len(gwave_l)), twave_l/max(twave_l)-1.0)
    ax[0,0].legend(loc="upper right")
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
