import numpy as np
import scipy.ndimage as ndimage
import scipy.signal as sig
import scipy.interpolate as intp
import scipy.integrate as intg
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

# Apply uniform filter to average out each point with neighbours
def moving_average(a, n=5):
    return ndimage.uniform_filter(a,n)

"""
# Initial PSD attempts, works but produces jagged result that gave worse
# results of matched filter.
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
"""

# Take PSDs using welch's method. Produces smoother output
def psd(signal, dt):
    nperseg = 4 * 4096
    # Take rfft
    freqs, psd = sig.welch(signal, 4096, window="blackman", nperseg=nperseg)
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
plt.title("Comparing Averages")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Noise PSD (m^2/Hz)")
plt.show(block=True)

# 3 looks pretty good as it avoids flat tops of peaks but also
# removes jagged spikes and a bit of noise.
h_spect = moving_average(h_spect,3)
l_spect = moving_average(l_spect,3)
h_itp = intp.interp1d(freq, h_spect)
l_itp = intp.interp1d(freq, l_spect)

# Plot the noise model
plt.figure()
plt.loglog(freq, np.sqrt(h_spect), label="Hanford")
plt.loglog(freq, np.sqrt(l_spect), label="Livingston")
plt.title("Ligo Noise ASD")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Noise ASD (m/Sqrt(Hz))")
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
    white_strain = np.fft.ifft(spectr)

    # apply bandpass from 20 to 2000 which LIGO indicates as the usable band
    bb,ba = sig.butter(5, np.array([20,2000]) * 2 * dt, btype="band")
    norm = np.sqrt((2000-20)*2*dt)
    return sig.lfilter(bb,ba,white_strain) / norm

def match_filter(strain, template, noise, dt):
    window = sig.tukey(strain.size, alpha=1./8)
    # Take the fft of the data and template
    strain_spect = np.fft.fft(strain*window) * dt
    temp_spect   = np.fft.fft(template*window) * dt
    # Get Frequencies
    freq = np.fft.fftfreq(len(window), dt)
    # Get freq spacing
    df = freq[1]-freq[0]
    # Get interpolated noise at each frequency
    noise_itp = noise(np.abs(freq))

    # Do the matched filter
    optimal = strain_spect * temp_spect.conjugate() / noise_itp
    # Integrate the filter and find point along f-axis
    # where integral is equal to half of the total
    # This is the frequency at below and above which half the
    # weight is stored.
    int = intg.cumtrapz(np.abs(optimal), dx=df, initial=0)
    mid_idx = np.argmin(np.abs(int - max(int)/2))
    # Inverse fourier transform and scale to go back to time.
    optimal_time = 2 * np.fft.ifft(optimal) / dt

    # Match filter against template for noise estimates
    sig_opt = temp_spect * temp_spect.conjugate() / noise_itp
    
    # Scatter in signal for noise estimate
    sigma_data = np.std(np.abs(optimal_time))
    max_data = np.max(np.abs(optimal_time))
    SNR_data = (max_data-np.mean(np.abs(optimal_time)))/sigma_data

    # Calculate sigma for getting signal to noise. Include normalization
    sigmasq = np.sum(sig_opt) * df
    sigma = np.sqrt(np.abs(sigmasq))
    print("\tSigma = ",sigma)

    SNR = optimal_time / sigma
    return SNR, SNR_data, freq[mid_idx]


# Loop through each dataset to find events
with open("data/BBH_events_v3.json") as json_file:
    json_data = json.load(json_file)

for event in json_data.keys():
    event_json = json_data[event]
    print('reading event', event_json['name'])
    
    # Load strain data
    strain_h,dt_h,utc_h=read_file(dataFolder + "/" + event_json['fn_H1'])
    strain_l,dt_l,utc_l=read_file(dataFolder + "/" + event_json['fn_L1'])
    
    # Holding variable
    abs_max_h = 0
    abs_max_l = 0

    fig,ax = plt.subplots(4,2)
    # Lets pretend we don't know which template to apply
    for idx, template_name in enumerate(templates):
        print('\n\tTrying template ', template_name)
        # Load polarizations of template
        t_p,t_c=read_template(template_name)
        # Make complex template
        temp = (t_p + t_c * 1.0j)
        
        # Match filter Hanford
        SNR_h, SNR_data_h, mid_freq_h = match_filter(strain_h, np.copy(temp),
                                                      h_itp, dt_h)
        # Shift output to put t=0 at start
        SNR_h = np.fft.fftshift(SNR_h)
        # Match filter Livingston
        SNR_l, SNR_data_l, mid_freq_l = match_filter(strain_l, np.copy(temp),
                                                      l_itp, dt_l)
        # Shift ouput to put t=0 at start
        SNR_l = np.fft.fftshift(SNR_l)
        
        # Find index of max SNR
        idxmax_h = np.argmax(np.abs(SNR_h))
        idxmax_l = np.argmax(np.abs(SNR_l))
        
        # Get max value of SNR
        max_h = np.max(np.abs(SNR_h))
        max_l = np.max(np.abs(SNR_l))

        # Naively pull out uncertainty in timing by
        # taking the times defining the FWHM of each SNR peak
        # from here we can estimate the time uncertainty
        # np.argsort gives the indices that would sort the array
        # see the first two are the closest to max/2
        fwhm_h = np.abs(np.diff(np.argsort(np.abs(np.abs(SNR_h) - max_h/2))[:2]))[0] * dt
        fwhm_l = np.abs(np.diff(np.argsort(np.abs(np.abs(SNR_l) - max_l/2))[:2]))[0] * dt

        # Store the information on the best template fit
        if max_h > abs_max_h and max_l > abs_max_l:
            abs_max_h = max_h
            abs_max_l = max_l
            time_h = idxmax_h * dt
            time_l = idxmax_l * dt
            # Calc sigma from fwhm
            sigma_h = fwhm_h/2.355
            sigma_l = fwhm_l/2.355
            SNR_ex_max_h = SNR_data_h
            SNR_ex_max_l = SNR_data_l
            freq_max_h = mid_freq_h
            freq_max_l = mid_freq_l
            max_name = template_name

        # Whiten the strain for plotting
        wstrain_h = whiten(strain_h, h_itp, dt_h)
        wstrain_l = whiten(strain_l, l_itp, dt_l)

        # Phase shift the template based on SNR max, c/abs(c) = exp(-1j arg(c))
        phased_h = whiten(temp * SNR_h[idxmax_h]/abs(SNR_h[idxmax_h]),
                          h_itp, dt_h)
        phased_l = whiten(temp * SNR_l[idxmax_l]/abs(SNR_l[idxmax_l]),
                          l_itp, dt_h)
        # Take the real part for plotting
        wtemp_h = np.real(phased_h)
        wtemp_l = np.real(phased_l)

        # Plot the Matched Filter SNR output
        ax[idx,0].plot(np.arange(len(SNR_h)) * dt_h,np.abs(SNR_h), label="H")
        ax[idx,0].plot(np.arange(len(SNR_l)) * dt_l,np.abs(SNR_l), label="L")
        ax[idx,0].set_ylabel(template_name.split("/")[2].split("_")[0])

        # Only plot +/- 0.1 seconds around the detected event
        offset = int(round(0.1/dt))

        # Take the whitened strain data around the event max
        gwave_h = np.real(wstrain_h[idxmax_h-offset:idxmax_h+offset])
        # Take the whitened template data around its peak which is in the middle
        twave_h = np.real(wtemp_h[len(wtemp_h)//2 - offset:
                                  len(wtemp_h)//2 + offset])
        # Plot the normalized whitend strain data, shifted up for H detector
        ax[idx,1].plot(np.linspace(-0.1,0.1,len(gwave_h)),
                                   gwave_h/max(gwave_h)+1.0)

        # Take the whitened strain data around the event max
        gwave_l = np.real(wstrain_l[idxmax_l-offset:idxmax_l+offset])
        # Take the whitened template data around its peak which is in the middle
        twave_l = np.real(wtemp_l[len(wtemp_l)//2 - offset:
                                  len(wtemp_l)//2 + offset])
        # Plot the normalized whitend strain data, shifted down for H detector
        ax[idx,1].plot(np.linspace(-0.1,0.1,len(gwave_l)),
                       gwave_l/max(gwave_l)-1.0)

        # Overlay the normalized whitened templates, shifted for the respective
        # detectors
        ax[idx,1].plot(np.linspace(-0.1,0.1,len(gwave_h)),
                       twave_h/max(twave_l)+1.0)
        ax[idx,1].plot(np.linspace(-0.1,0.1,len(gwave_l)),
                       twave_l/max(twave_l)-1.0)
        ax[idx,1].set_xlim([-0.1,0.1])

    # Print Results
    print("\tBest Template: %s" % max_name)
    print("\t\tHanford:    SNR = %f @ %f +/- %f s" % (abs_max_h, time_h,sigma_h))
    print("\t\t\tMid Freq = %f, Expected SNR = %f" % (freq_max_h, SNR_ex_max_h))
    print("\t\tLivingston: SNR = %f @ %f +/- %f s" % (abs_max_l, time_l,sigma_l))
    print("\t\t\tMid Freq = %f, Expected SNR = %f" % (freq_max_l, SNR_ex_max_l))
    print("\n")

    # Annotate the Plot
    ax[0,0].set_title("Matched Filter Output (SNR)")
    ax[0,1].set_title("Normalized Strain Data (A.U.)")
    ax[1,0].legend(loc="upper right")
    ax[3,0].set_xlabel("Time")
    ax[3,1].set_xlabel("Time from SNR peak")
    plt.show(block=True)
