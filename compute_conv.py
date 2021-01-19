import numpy as np
import itertools
from typing import List, Dict, Optional, Any

from make_time_shift import unroll_kernel, unroll_image, generate_time_shift_signal_batch

def simulate_time_delay(signal_batch:List[List[float]]) -> List[List[float]]:
	for i in range(len(signal_batch)):
		signal_batch[i] = signal_batch[i][i:] + signal_batch[i][:i]
	return signal_batch
	
def simulate_conv(signal_batch: List[List[float]], input_figure: List[List[float]], kernel: List[List[List[float]]]):
	channel_num = len(kernel)
	unrolled_kernel = np.array(unroll_kernel(kernel)).reshape(channel_num, -1)
	kernel_height = len(kernel[0])
	kernel_width = len(kernel[0][0])

	# simulate we only take the first group of signals of kernel1 length
	signal_batch = signal_batch[:len(signal_batch) // len(kernel)]
	conv_res = []
	for k in range(len(unrolled_kernel)):
		for i in range(0, len(signal_batch[0]), 2):
			cell_res = 0.0
			for j in range(len(unrolled_kernel[0])):
				cell_res += signal_batch[j][i] * unrolled_kernel[k][j] 
			conv_res.append(cell_res)

	num_results_per_line = len(input_figure[0]) - kernel_width + 1
	conv_res = np.array(conv_res).reshape(channel_num, len(input_figure) // kernel_height, -1)

	conv_res = conv_res[:, :, :num_results_per_line]

	return conv_res
	
if __name__ == "__main__":
	kernel = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
	one_d_kernel = unroll_kernel(kernel)

	input_figure = np.array([1.0, 0.0] * 8).reshape(4, 4)
	unrolled_input = unroll_image(input_figure, kernel)

	signal_batch = generate_time_shift_signal_batch(kernel, unrolled_input, 0.001)

	shifted_signals = simulate_time_delay(signal_batch)

	conv_results = simulate_conv(signal_batch, input_figure, kernel)
	print(conv_results)

	
	