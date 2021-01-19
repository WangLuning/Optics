import itertools
import copy
import matplotlib.pyplot as plt
import matplotlib._color_data as mcd
from matplotlib.patches import Rectangle
from typing import List, Dict, Optional, Any

from generate_comb import *

overlap = sorted({name for name in mcd.CSS4_COLORS
           if "xkcd:" + name in mcd.XKCD_COLORS}, reverse = True)

# flatten 3d to 1d kernel
def unroll_kernel(kernel:List[List[List[float]]]) -> List[float]:
	channels, height, width = len(kernel), len(kernel[0]), len(kernel[0][0])
	unrolled_kernel = []
	for c in range(channels):
		for j in range(width):
			for i in range(height):
				unrolled_kernel.append(kernel[c][i][j])

	return unrolled_kernel


def unroll_image(image:List[List[float]], kernel:List[List[List[float]]]):
	kernel_row = len(kernel[0])
	kernel_col = len(kernel[0][0])
	image_row = len(image)
	image_col = len(image[0])
	unrolled_image = []
	
	for i in range(0, image_row // kernel_row * kernel_row, kernel_row):
		for j in range(image_col):
			for k in range(kernel_row):
				unrolled_image.append(image[i + k][j])
	return unrolled_image
			

def smf(tmin_num, tmax_num, tmax, ht, alpha, delta0, beta_group, omega, uu, theta, E_in, tr, h, check_soliton = False):
	'''
	removed for confidential purpose
	'''

	return uu, spect, lIW


# if signal > 0.5 then round it to 1 else 0
def threshold_decider(truncated_signal):
	if max(truncated_signal) > 0.5:
		truncated_signal = 1.0 / (float(max(truncated_signal))) * truncated_signal
	signal = [int(round(x)) for x in truncated_signal]
	return signal


# input N*N image, output a N*N smf signal after threshold
def generate_time_shift_signal(unrolled_input:List[float], smf_distance:float) -> List[int]:
	figure_size = len(unrolled_input)
	# note that it is spectral that we modulate on
	unrolled_input = [0.0] * (ntau // 2) + unrolled_input + [0.0] * (ntau // 2 - figure_size)
	# change the modulated spectral back to time domain
	uu = np.fft.fftshift(np.fft.fft(np.fft.fftshift(unrolled_input)))

	# smf with no amplifier
	smf_signal, smf_spect, smf_lIW = smf(0.0, smf_distance, tmax, ht, alpha, delta0, beta_group, omega, uu, theta, 0.0, tr, h)

	truncated_signal = [abs(s) for s in smf_spect][ntau//2 : ntau//2 + figure_size]
	decided_signal = threshold_decider(np.array(truncated_signal))

	return decided_signal


def generate_time_shift_signal_batch(kernel:List[List[List[float]]], input:List[float], smf_distance:float) -> List[List[float]]:
	signal_batch = []
	unrolled_kernel = unroll_kernel(kernel)
	num_shift = len(unrolled_kernel)

	for i in range(num_shift):
		input = generate_time_shift_signal(input, smf_distance)
		signal_batch.append(copy.deepcopy(input))
	return signal_batch


def show_original_image(figure_list:List[List[float]]):
	fig, ax = plt.subplots()
	
	sig1 = mcd.CSS4_COLORS[overlap[0]]
	sig0 = mcd.CSS4_COLORS[overlap[1]]
	ax.plot([0, len(figure_list[0]) + 2],[0, len(figure_list) + 2])

	for i in range(len(figure_list)):
		for j in range(len(figure_list[i])):
			if figure_list[i][j] == 0.0:
				ax.add_patch(Rectangle((j, i), 1, 1, facecolor=sig0, linewidth=1, edgecolor='b'))
			else:
				ax.add_patch(Rectangle((j, i), 1, 1, facecolor=sig1, linewidth=1, edgecolor='b'))
				
	plt.show()


def show_time_shift_results(figure_list:List[List[int]], one_d_kernel:List[float]):
	# figure_list: (1d_kernel_size, unrolled_figure_size)

	fig, ax = plt.subplots()
	
	sig1 = mcd.CSS4_COLORS[overlap[0]]
	sig0 = mcd.CSS4_COLORS[overlap[1]]
	ax.plot([0, len(figure_list[0]) + len(one_d_kernel)],[0, len(figure_list) + 2])

	for i in range(len(figure_list)):
		for j in range(len(figure_list[i])):
			if figure_list[i][j] == 0.0:
				ax.add_patch(Rectangle((j + len(figure_list) - i, i), 1, 1, facecolor=sig0, linewidth=1, edgecolor='b'))
			else:
				ax.add_patch(Rectangle((j + len(figure_list) - i, i), 1, 1, facecolor=sig1, linewidth=1, edgecolor='b'))
				
	plt.show()

if __name__ == "__main__":
	# unroll multi-channel kernels to 1d
	# we do not compute kernel related, but just to know how many time shifts we need
	kernel = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
	one_d_kernel = unroll_kernel(kernel)

	# suppose here we have a figure of 4*4 and unroll it
	input_figure = np.array([1.0, 0.0] * 8).reshape(4, 4)
	show_original_image(input_figure)

	# TODO: figure should be unrolled according to kernel size
	unrolled_input = unroll_image(input_figure, kernel)

	signal_batch = generate_time_shift_signal_batch(kernel, unrolled_input, 0.001)

	show_time_shift_results(signal_batch, one_d_kernel)

