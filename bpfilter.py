# https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html #

import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz
	
from scipy.signal import butter, lfilter
from constant_variables import *
from plot_helper import plot_graph

if os.path.isfile('finetuned_params.pkl'):
	with open('finetuned_params.pkl', 'rb') as f:
		uu, spect, lIW = pickle.load(f)
			
def butter_bandpass(lowcut, highcut, fs, order=5):
	nyq = 0.5 * fs
	low = lowcut / nyq
	high = highcut / nyq
	b, a = butter(order, [low, high], btype='band')
	return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
	b, a = butter_bandpass(lowcut, highcut, fs, order=order)
	y = lfilter(b, a, data)
	return y


def design_bpf(fs, lowcut, highcut):
	# Plot the frequency response for a few different orders.
	plt.figure(1)
	plt.clf()
	for order in [3, 6, 9]:
		b, a = butter_bandpass(lowcut, highcut, fs, order=order)
		w, h = freqz(b, a, worN=2000)

		x_axis = (fs * 0.5 / np.pi) * w
		y_axis = abs(h)
		plt.xlim(1.9e2, 2.1e2)
		plt.plot(x_axis, y_axis, label="order = %d" % order)
	plt.xlabel('Frequency (THz)')
	plt.ylabel('Gain')
	plt.grid(True)
	plt.legend(loc='best')
	plt.show()


def run_filter():
	# Sample rate and desired cutoff frequencies (in THz).
	# our filter is about 200THz
	fs = ntau
	lowcut = c / 1500
	highcut = c / 1494

	design_bpf(fs, lowcut, highcut)

	# apply the filter on the OFC
	t = np.linspace(0, dtau, int(Tmax), endpoint=False)
	uu_filtered = butter_bandpass_filter(uu, lowcut, highcut, fs, order=3)

	# the original signal is
	plt.plot(tau,uu)
	plt.title('Time Domain')
	plt.xlabel('Time tau (ps)')
	plt.ylabel('Power(W)')

	plt.ylim([-0.02, 0.02])
	plt.plot(tau, uu_filtered, label='Filtered signal (%g Hz)' % f0)
	plt.grid(True)
	plt.axis('tight')
	plt.legend(loc='upper left')
	plt.title('OFC signal through BPF in time domain')
	plt.show()

	spect_filtered=np.fft.fftshift(np.fft.ifft(np.fft.fftshift(uu_filtered)))
	lIW_filtered = np.array([10 * math.log10(abs(s)**2) if s > 0.0 else float('-inf') for s in spect_filtered])
	plt.stem(lambdas,lIW_filtered, bottom = -80)
	plt.title('Frequency Domain')
	plt.xlabel('Wavelength lambda (nm)')
	plt.ylabel('Spectrum(dB)')
	plt.xlim([1200, 2400])
	plt.ylim([-80, 20])
	plt.show()


if __name__ == "__main__":
	run_filter()

	# in run() we get the filtered signal from OFC
	# below is simple selection of spectral lines for simulation purpose
	find_index = np.intersect1d(np.argwhere(lambdas > 1494), np.argwhere(lambdas < 1500))
	print(find_index)
	spect_lines = [lIW[idx] for idx in find_index]
