import numpy as np
# Used by crocker_signals_timing_fixed


# This file meant to fix whatever issues in crocker_signals_timing
def linear_interpolate_trigger(time_bins, waveform, baseline, f=np.array([0.2])):
    """Assumes positive polarity signals"""
    wf = waveform - baseline
    t = time_bins
    max_sig_fv = f * np.max(wf)

    ati = np.argmax(np.sign(wf - max_sig_fv))  # (a)fter (t)rigger (i)ndex
    m = (wf[ati] - wf[ati-1]) / (t[ati]-t[ati-1])  # slope
    interp_t = t[ati-1] + ((max_sig_fv - wf[ati-1])/m)

    return interp_t, max_sig_fv + baseline  # add back baseline for plotting


def leading_edge_trigger(time_bins, waveform, baseline, thr=0.1):
    """Leading edge trigger above baseline with threshold in V"""
    wf = waveform - baseline
    return time_bins[np.argmax(wf > thr)], thr + baseline


def rise_time_points(time_bins, waveform, baseline, f=np.array([0.1, 0.2, 0.9])):
    """Same as linear interpolate trigger but returns multiple thresholds and the maximum relative amplitude.
     Used for t0 rise time analysis and triggers."""
    wf = waveform - baseline
    t = time_bins

    interp_trgs = np.zeros_like(f)

    max_sig = np.max(wf)
    for i, fract in enumerate(f):
        max_sig_fv = f * max_sig
        ati = np.argmax(np.sign(wf - max_sig_fv))  # (a)fter (t)rigger (i)ndex
        m = (wf[ati] - wf[ati - 1]) / (t[ati] - t[ati - 1])  # slope
        interp_trgs[i] = t[ati - 1] + ((max_sig_fv - wf[ati - 1]) / m)

    return interp_trgs, max_sig  # relative max above baseline


def correlate_pulse_trains(t1, t2):
    """Correlates t1 values to nearest t2. T1 and T2 do not need to be the same size. T1 is tf, T2 is t0"""
    time_pairs = t1 - t2[:, np.newaxis]  # first t1 value - t2 values are in first row, second t1 value - t2 in 2nd, ...
    closest_t_idx = np.argmin(np.abs(time_pairs), axis=np.argmax(time_pairs.shape).item())
    try:
        if np.argmax(time_pairs.shape).item():
            del_t = time_pairs[np.arange(np.min(time_pairs.shape)), closest_t_idx]
        else:
            del_t = time_pairs[closest_t_idx, np.arange(np.min(time_pairs.shape))]
    except IndexError:
        print("argmax axis: ", np.argmax(time_pairs.shape).item())
        print("Error")
        print("closest_t_idx: ", np.argmin(np.abs(time_pairs), axis=np.argmax(time_pairs.shape).item()))
        print("time_pairs: ", time_pairs)
        print("time_pairs shape: ", time_pairs.shape)
    return del_t  # hopefully the closest pairs of values