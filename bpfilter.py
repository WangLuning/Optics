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
		x_axis = x_axis[1550:1650]
		y_axis = abs(h)[1550:1650]
		plt.plot(x_axis, y_axis, label="order = %d" % order)
	plt.xlabel('Frequency (Hz)')
	plt.ylabel('Gain')
	plt.grid(True)
	plt.legend(loc='best')
	plt.show()


def run_filter():
	# Sample rate and desired cutoff frequencies (in Hz).
	# our filter is about 200THz
	fs = 5e14
	lowcut = c / 1500 * 1e12
	highcut = c / 1494 * 1e12

	design_bpf(fs, lowcut, highcut)

	# apply the filter on the OFC
	T = 1e-11 *4096 / 5000
	nsamples = int(T * fs) + 1
	t = np.linspace(0, T, nsamples, endpoint=False)
	uu_filtered = butter_bandpass_filter(uu, lowcut, highcut, fs, order=3)
	plt.ylim([-0.02, 0.02])
	plt.plot(t, uu_filtered, label='Filtered signal (%g Hz)' % f0)
	plt.xlabel('time (seconds)')
	plt.grid(True)
	plt.axis('tight')
	plt.legend(loc='upper left')
	plt.title('OFC signal through BPF in time domain')
	plt.show()


if __name__ == "__main__":
	run()

	# in run() we get the filtered signal from OFC
	# below is simple selection of spectral lines for simulation purpose
	find_index = np.intersect1d(np.argwhere(lambdas > 1494), np.argwhere(lambdas < 1500))
	print(find_index)
	spect_lines = [lIW[idx] for idx in find_index]
