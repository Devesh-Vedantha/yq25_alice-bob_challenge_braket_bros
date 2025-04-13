from functions_lib import *
import dynamiqs as dq
from scipy.integrate import dblquad
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator
import numpy as np

# input is noisy wigner function
file_path = "/common/home/sps247/code/yq25_alice-bob_challenge_braket_bros/data/synthetic/noisy_wigner_0.pickle"
wig_noisy = load_data_from_pickle(file_path)
# p_new = aaron_1b(Z)
# outputs the perfect wigner function
wig_perf = dq.wigner(state=p_new)
def estimate_a(wig_noisy, wig_perf):
    num = np.sum(wig_noisy * wig_perf)
    denom = np.sum(wig_perf ** 2)
    a = num / denom
    return a
def estimate_b(wig_noisy, wig_perfect, a, n_simulations=1000):
    # estimate b using edges of phase space
    top_edge = wig_noisy[0, :]
    bottom_edge = wig_noisy[-1, :]
    left_edge = wig_noisy[:, 0]
    right_edge = wig_noisy[:, -1]

    # combine edge values
    edges = np.concatenate((top_edge, bottom_edge, left_edge, right_edge))

    # estimate b as mean of edge values
    b_initial = np.mean(edges)

    # want to minimize b
    b_range = 0.1 # arbitrarily set, adjust as needed
    b_min = b_initial - b_range
    b_max = b_initial + b_range

    # use monte carlo simulation to minimize value of b
    smallest_b = None
    min_noise = float('inf') # want to minimize noise, so initialize to infinity

    for i in range(n_simulations):
        b = np.random.uniform(b_min, b_max)
        wig_simulated = a * wig_perf + b # simulate noisy wigner function based on random b value
        noise = np.mean((wig_simulated - wig_noisy) ** 2) # calculate noise as mean squared error

        # update smallest b if noise is smaller than current minimum
        if noise < min_noise:
            min_noise = noise
            smallest_b = b
    return smallest_b

def correct_affine_distortion(wig_noisy, x, p):
    # create interpolator to check integral value
    x = np.linspace(bottom_edge, top_edge, wig_noisy.shape[0])
    p = np.linspace(left_edge, right_edge, wig_noisy.shape[1])
    interpolator = RegularGridInterpolator((x, p), wig_noisy)

    def interpolate(x, p):
        return interpolator((x, p))
    
    integral, error = dblquad(interpolate, left_edge, right_edge, lambda x: bottom_edge, lambda x: top_edge)

    # check normalization
    if not np.isclose(integral, 1):
        raise ValueError("Wigner function is not normalized.")
    
    else:
        # correct wigner function
        wig_corrected = (1 / a) * (wig_noisy - b)

    
def gauss_filter(wig_corrected, sigma):
    wig_smoothed = gaussian_filter(wig_corrected, sigma=0.1) # sigma arbitrarily set, may be changed
    return wig_smoothed
