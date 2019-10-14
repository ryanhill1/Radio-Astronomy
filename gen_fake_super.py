# Ryan Hill
# Cornell University, Dept. of Astronomy
# Program to generate and superimpose 
# fake single pulse and RFI on dynamic spectra

import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import matplotlib as mpl
import random

# Specify directory to which output fake DS will be written.
OUTPUT_DIR = '/Users/ryanhill/Desktop/research/pipeline/test_data_snr/'
PLOTS_DIR = '/Users/ryanhill/Desktop/research/pipeline/'

########## INPUT PARAMETERS ##########
# Input parameters for the dynamic spectrum.
n_time_bins = 128 # No. of time bins in the dynamic spectrum
n_freq_bins = 128 # No. of spectral channels
time_resol = 6.4e-5 # Time resolution of data (s)
freq_start = 1.15 # Frequency (GHz) at lower edge of bandpass.
freq_stop = 1.73 # Frequency (GHz) at upper edge of bandpass.

N_RFI = 10 # No. of RFI candidates (DM=0 pc cm^{-3}) to simulate.
N_pulses = 0 # No. of dispersed single pulses to simulate.
N_PULSE_AND_RFI = 0 # No. of graphs with single pulse and RFI 
# Dispersion is performed in accordance with the cold plasma dispersion relation.

# Sample pulse DMs from a uniform distribution between limits DM_min and DM_max.
DM_min = 1.0 # Minimum DM (pc cm^{-3})
DM_max = 1000.0 # Maximum DM (pc cm^{-3})

# Sample signal FWHM from a uniform distribution between pulse_bw_min and pulse_bw_max.
# BROADBAND
broadband_min = 0.1 # Minimum pulse bandwidth (GHz)
broadband_max = 0.5 # Maximum pulse bandwidth (GHz)
# NARROWBAND
narrowband_min = 0.0001 # Minimum pulse bandwidth (GHz)
narrowband_max = 0.01 # Maximum pulse bandwidth (GHz)

# Range of temporal widths of RFI.
# SHORT-LIVED
shortlived_min = 1e-4 # Minimum temporal width (s) of an RFI signal.
shortlived_max = 4e-3 # Largest allowed temporal width (s) of an RFI.
# LONG_LIVED
longlived_min = 1e-2
longlived_max = 1e-1

# Range of pulse FWHMs along time axis.
pulse_time_FWHM_min = 1e-4 # Minimum temporal width (s) of a pulse.
pulse_time_FWHM_max = 1e-2 # Largest allowed pulse temporal width (s).

# Range of pulse bandwidth FWHM
pulse_bw_min = 0.1
pulse_bw_max = 1.0

# SNR of pulses. Sample signal SNRs from a uniform distribution between these limits.
SNR_min = 5.0 # Minimum signal SNR
SNR_max = 15.0 # Maximum signal SNR

########## FUNCTION DEFINITIONS ##########
# Specify RFI category and assign values
def rfi_parameters(category):
	if category == 0: #llbb
		parameters = [broadband_min, broadband_max, longlived_min, longlived_max]
	elif category == 1: #llnb
		parameters = [narrowband_min, narrowband_max, longlived_min, longlived_max]
	elif category == 2: #slbb
		parameters = [broadband_min, broadband_max, shortlived_min, shortlived_max]
	else: #slnb
		parameters = [narrowband_min, narrowband_max, shortlived_min, shortlived_max]
	return parameters

# Sample from a log-uniform distribution.
def loguniform(low_limit,high_limit,n_samples):
	log_low_limit = np.log(low_limit)
	log_high_limit = np.log(high_limit)
	log_samples = np.random.uniform(log_low_limit,log_high_limit,n_samples)
	samples = np.exp(log_samples)
	return samples
	
# Simulate Gaussian random noise with zero mean and unit variance.
def noise_std_normal(n_freq_bins,n_time_bins):
	noise = np.random.randn(n_freq_bins,n_time_bins)
	return noise
	
# Simulate a dispersed pulse (Gaussian profile along frequency and time) (no noise)
def simulate_pulse(f_center,t_center,FWHM_f,FWHM_t,SNR,n_time_bins,
n_freq_bins,freq_array,time_array,DM):
	pulse = np.zeros((n_freq_bins, n_time_bins))
	# Convert supplied FWHM widths along frequency and time axes to 1/e widths.
	sigma_f = FWHM_f/np.sqrt(8*np.log(2))
	sigma_t = FWHM_t/np.sqrt(8*np.log(2))
	# Simulate dispersed pulse.
	for i in range(n_freq_bins):
		nu = freq_array[i]
		t_shift = 4.15*DM*(nu**-2. - f_center**-2.)
		for j in range(n_time_bins):
			t = time_array[j]
			pulse[i,j] = SNR*np.exp(-0.5*((nu - f_center)/sigma_f)**2)*np.exp(-0.5*
			((t - t_center-t_shift)/sigma_t)**2)
	return pulse
	
# Plot dynamic spectrum and show it during code execution.
def plot_ds_show(dyn_spectrum,freq_array,time_array,SNR,DM,savefile,PLOTS_DIR):
	# Boundaries of the dynamic spectrum.
	extent_ds = [time_array[0],time_array[-1],freq_array[0],freq_array[-1]]	
	plt.imshow(dyn_spectrum,origin='lower',interpolation='None',aspect='auto',extent=extent_ds)
	plt.xlabel('Time (ms)',fontsize=14)
	plt.ylabel('Radio frequency (GHz)',fontsize=14)
	plt.title('S/N = %.1f, DM = %.1f pc cm$^{-3}$'% (SNR,DM))
	h = plt.colorbar()
	h.set_label('Flux density (arbitrary units)',fontsize=14)
	plt.savefig(PLOTS_DIR+savefile+'.png')
	plt.show()
	plt.close()	

########## DERIVED QUANTITIES ##########
tot_time = time_resol*n_time_bins*1e3 # Total duration (ms) of data set.
bandwidth = freq_start - freq_stop # Total bandwidth (GHz) of data set.
chan_bandwidth = (freq_stop - freq_start)*1e3/n_freq_bins # Channel bandwidth (MHz)
# Array of frequencies corresponding to spectral channels.
freq_array = np.linspace(freq_start,freq_stop,n_freq_bins) # GHz
# Array of time stamps for each pixel of the dynamic spectrum.
time_array = np.linspace(0,tot_time,n_time_bins) # ms

# Generate properties of dispersed pulses.
if (N_pulses > 0 or N_PULSE_AND_RFI > 0):
  # N_pulses = N_PULSE_AND_RFI
	# Center frequencies and times of pulses.
  f_center_pulses = np.random.uniform(freq_start,freq_stop,N_pulses) # GHz
  t_center_pulses = np.random.uniform(0.,tot_time,N_pulses) # ms
	# FWHM of pulse signal
  FWHM_freq_pulses = np.random.uniform(pulse_bw_min,pulse_bw_max,N_pulses) # GHz
  FWHM_time_pulses = loguniform(pulse_time_FWHM_min,pulse_time_FWHM_max,N_pulses)*1e3 # ms
  SNR_pulses = np.random.uniform(SNR_min,SNR_max,N_pulses)
	# Generate pulse DMs.
  pulse_DMs = loguniform(DM_min,DM_max,N_pulses) # pc cm^{-3}
	
# Generate properties of RFI.
if (N_RFI > 0 or N_PULSE_AND_RFI > 0):
  # N_RFI = N_PULSE_AND_RFI
	# Center frequencies and times of RFI signals.
  f_center_RFI = np.random.uniform(freq_start,freq_stop,N_RFI) # GHz
  t_center_RFI = np.random.uniform(0.,tot_time,N_RFI) # ms
  SNR_RFI = np.random.uniform(SNR_min,SNR_max,N_RFI)
  DM_RFI = 0.0

# FWHM of RFI signals
def FWHM_freq_and_time(category):
	# category = random.randint(0,4)
	parameters = rfi_parameters(category)
	pulse_bw_min = parameters[0]
	pulse_bw_max = parameters[1]
	RFI_time_FWHM_min = parameters[2]
	RFI_time_FWHM_max = parameters[3]
	FWHM_freq_RFI = np.random.uniform(pulse_bw_min,pulse_bw_max,1) # GHz
	FWHM_time_RFI = loguniform(RFI_time_FWHM_min,RFI_time_FWHM_max,1)*1e3 # ms
	return (FWHM_freq_RFI[0], FWHM_time_RFI[0], category)

########## CREATE PLOT(S) ##########
# N_RFI = 0
# N_pulses = 0
for i in range(N_RFI):
  noise = noise_std_normal(n_freq_bins,n_time_bins) # abstracting noise array
  ftc = FWHM_freq_and_time(random.randint(0,3))
  rfi_pulse = simulate_pulse(f_center_RFI[i],t_center_RFI[i],ftc[0],ftc[1],
	SNR_RFI[i],n_time_bins,n_freq_bins,freq_array,time_array,DM_RFI)
  ds_RFI = noise + rfi_pulse
  SNR = SNR_RFI[i]
  choose_noise = random.randint(0,4)
  if choose_noise == 4:
    ds_RFI = noise
    SNR = -1
    ftc = (0,0,4)
  ds_RFI = np.array([ds_RFI, [ftc[2]], [SNR]]) # add category when creating test data
  savefile = 'test_rfi_'+str(i)
  # plot_ds_show(ds_RFI,freq_array,time_array,SNR_RFI[i],0.0,savefile,PLOTS_DIR)
  print('RFI no.: %d'% (i+1))
  np.save(OUTPUT_DIR+savefile,ds_RFI)

for i in range(N_pulses):
  noise = noise_std_normal(n_freq_bins,n_time_bins) # abstracting noise array
  pulse = simulate_pulse(f_center_pulses[i],t_center_pulses[i],
	FWHM_freq_pulses[i],FWHM_time_pulses[i],SNR_pulses[i],n_time_bins,
	n_freq_bins,freq_array,time_array,pulse_DMs[i])
  ds_pulse = noise + pulse	
  savefile = 'test_pulse_'+str(i)
  plot_ds_show(ds_pulse,freq_array,time_array,SNR_pulses[i],
	pulse_DMs[i],savefile,PLOTS_DIR)
  print('Pulse no.: %d'% (i+1))
  np.save(OUTPUT_DIR+savefile,ds_pulse)

for i in range(N_PULSE_AND_RFI):
  noise = noise_std_normal(n_freq_bins,n_time_bins) # abstracting noise array
  ftc = FWHM_freq_and_time(3)
  rfi = simulate_pulse(f_center_RFI[i],t_center_RFI[i],ftc[0],ftc[1],
	SNR_RFI[i],n_time_bins,n_freq_bins,freq_array,time_array,DM_RFI)
  pulse = simulate_pulse(f_center_pulses[i],t_center_pulses[i],
	FWHM_freq_pulses[i],FWHM_time_pulses[i],SNR_pulses[i],
	n_time_bins,n_freq_bins,freq_array,time_array,pulse_DMs[i])
  ds_rfi_and_pulse = noise + rfi + pulse
	# ds_RFI = np.array([ds_RFI, [ftc[2]]]) # add category when creating test data
  savefile = 'TYPE_and_pulse'+str(i)
  plot_ds_show(ds_rfi_and_pulse,freq_array,time_array,SNR_RFI[i],0.0,savefile,PLOTS_DIR)
  print('RFI no.: %d'% (i+1))
  np.save(OUTPUT_DIR+savefile,ds_rfi_and_pulse)