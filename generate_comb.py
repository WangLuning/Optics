import cmath
import pickle
import os

from plot_helper import plot_graph
from constant_variables import *

###################

uu = np.asarray([0.9 * (1 + 0.1 * math.exp(-(t/0.0314)**2)) for t in tau])

spect = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(uu))) * (ntau*dtau)/math.sqrt(2*pi)
lIW = np.array([10 * math.log10(abs(s)**2) if s > 0.0 else float('-inf') for s in spect])

beta_group = (0, 0, beta2, beta3, beta4, beta5, beta6, beta7, beta8)

#plot_graph(tau, uu, lambdas, lIW)
###################

def LLE(tmin_num, tmax_num, tmax, ht, alpha, delta0, beta_group, omega, uu, theta, E_in, tr, h, check_soliton = False):
	'''
	removed for confidential purpose
	'''

	return uu, spect, lIW


def run_comb():
	first_step_tune = 8
	second_step_tune = 8.3

	if os.path.isfile('params.pkl'):
		with open('params.pkl', 'rb') as f:
			uu, spect, lIW = pickle.load(f)
			#plot_graph(tau, uu, lambdas, lIW)
	else:
		uu, spect, lIW = LLE(0.0, first_step_tune, tmax, ht, alpha, delta0, beta_group, omega, uu, theta, E_in, tr, h)
		with open('params.pkl', 'wb') as f:
			pickle.dump([uu, spect, lIW], f)

	# next step train
	if os.path.isfile('finetuned_params.pkl'):
		with open('finetuned_params.pkl', 'rb') as f:
			uu, spect, lIW = pickle.load(f)
			#plot_graph(tau, uu, lambdas, lIW)
	else:
		uu, spect, lIW = LLE(first_step_tune, second_step_tune, tmax, ht, alpha, delta0, beta_group, omega, uu, theta, E_in, tr, h, check_soliton = True)
		with open('finetuned_params.pkl', 'wb') as f:
			pickle.dump([uu, spect, lIW], f)

	plot_graph(tau, uu, lambdas, lIW)


if __name__ == "__main__":
	run_comb()
