from bpfilter import *
from compute_conv import *
from generate_comb import *
from make_time_shift import *

def main():
	# here explains the main process
	# generate optical frequency comb
	run_comb()
	# generate band pass filter to filter some lines
	run_filter()
	
	# the cnn kernels
	kernel = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
	one_d_kernel = unroll_kernel(kernel)

	# input image
	input_figure = np.array([1.0, 0.0] * 8).reshape(4, 4)
	show_original_image(input_figure)

	# unroll image to 1d and go through smf
	unrolled_input = unroll_image(input_figure, kernel)
	signal_batch = generate_time_shift_signal_batch(kernel, unrolled_input, 0.001)
	show_time_shift_results(signal_batch, one_d_kernel)

	# compute cnn
	shifted_signals = simulate_time_delay(signal_batch)
	conv_results = simulate_conv(signal_batch, input_figure, kernel)

	print(conv_results)

if __name__ == "__main__":
	main()