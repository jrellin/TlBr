import numpy as np


def gauss(x, a, mu, sig, bkg):
    #  a = amplitude
    #  mu = mean
    #  sigma = std. deviation
    #  bkg = (constant) background
    return a * np.exp(-(x - mu) ** 2 / (2 * sig ** 2)) + bkg


def gaussian_spread_guess(bins, histogram, percent_max=None):
    """Creates a guess for gaussian spread of provided bins and counts data.
    Defaults to 10 percent range on each side of the max to search for FWXM where X is percent max"""
    if percent_max >= 1:
        ValueError('Percent Max must be less than 1')
    if percent_max is None:
        percent_max = 0.5

    threshold = np.max(histogram) * percent_max
    start = np.argmax(histogram)  # start index
    # center = bins[start]  # center of gaussian
    left_bound = int(0.9 * np.argmax(histogram))
    right_bound = int(1.1 * np.argmax(histogram))

    right_side = histogram[start:right_bound]
    right_side_bins = bins[start:right_bound]
    left_side = histogram[left_bound:start]
    left_side_bins = bins[left_bound:start]

    r_idx = right_side.size - np.argmax(np.sign(right_side[::-1] - threshold)) - 1
    # distance from center to falling below threshold to the right
    l_idx = np.argmax(np.sign(left_side - threshold))  # same but to the left

    spread = (right_side_bins[r_idx] - left_side_bins[l_idx])/2   # FWXM
    return spread


def gauss_fit(x, y, percent_max=50):
    # x generally channels as bins, y generally counts
    from scipy.optimize import curve_fit

    guess_spread = gaussian_spread_guess(x, y, percent_max=percent_max)
    # p0 = [np.max(y), x[np.argmax(y)], 2 * guess_spread / 2.355]  # no bkg
    p0 = [np.max(y), x[np.argmax(y)], 2 * guess_spread/2.355, 0.01 * np.max(y)]
    yerr = np.sqrt(y).tolist()
    wgt = [1 / max(w, 0.001) for w in yerr]
    popt, pcov = curve_fit(gauss, x, y, p0=p0, sigma=wgt)
    return popt, pcov  # popt = amp, mu, sig, bkg
