import numpy as np


def linear_interpolate_trigger(time_bins, waveform, baseline, f=np.array([0.2]), ret_max_instead=False):
    """Assumes positive polarity signals. Trying to fix spike issue seen in lf2 p2 1200 skip trigger. Used in crocker_energy_timing.py"""
    wf = waveform - baseline
    t = time_bins
    max_sig_fv = f * np.max(wf)

    ati = np.argmax(wf) - np.argmin(wf[:np.argmax(wf)][::-1] >= max_sig_fv)
    m = (wf[ati] - wf[ati-1]) / (t[ati]-t[ati-1])  # slope
    if m != 0:
        interp_t = t[ati-1] + ((max_sig_fv - wf[ati-1])/m)
    else:
        interp_t = t[ati - 1]  # nothing to interp

    if ret_max_instead:
        return interp_t, np.max(wf) + baseline  # add back baseline for plotting
    return interp_t, max_sig_fv + baseline  # add back baseline for plotting

