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
    if m != 0:
        interp_t = t[ati-1] + ((max_sig_fv - wf[ati-1])/m)
    else:
        interp_t = t[ati - 1]  # nothing to interp

    return interp_t, max_sig_fv + baseline  # add back baseline for plotting


def linear_interpolate_trigger2(time_bins, waveform, baseline, f=np.array([0.2])):
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


def histogram_FWHM(time_bin_edges, counts, outside_pts=0):
    """Finds FHWM of a time histogram, need time bins and counts. Outside points refers to how many points outside
    before and after 0.5 crossings to include for linear fit"""
    from numpy.polynomial import polynomial as P
    time_bins = 0.5 * (time_bin_edges[1:] + time_bin_edges[:-1])

    op = int(outside_pts)  # alias
    h_max = np.max(counts)
    h_amax = np.amax(counts)
    lo_idx = h_amax - np.argmin(counts[:h_amax][::-1] >= 0.5 * h_max)  # just after threshold before max
    hi_idx = h_amax + np.argmin(counts[h_amax:] >= 0.5 * h_max)  # just before threshold past max

    rise_t = np.array([time_bins[lo_idx-1 - op], time_bins[lo_idx + op]])
    fall_t = np.array([time_bins[hi_idx - op], time_bins[hi_idx+1 + op]])

    lo_counts = np.array([counts[lo_idx-1 - op], counts[lo_idx + op]])  # points before and after threshold crossing
    hi_counts = np.array([counts[hi_idx - op], counts[hi_idx+1 + op]])

    r_fit = P.polyfit(rise_t, lo_counts, 1, w=1/np.sqrt(lo_counts))  # return [c0, c1] for y = c0 + x * c1
    f_fit = P.polyfit(fall_t, hi_counts, 1, w=1/np.sqrt(hi_counts))

    # y= c0 + c1 * x -> (y-c0)/c1 = x, where y = 0.5 * y_max

    r_t_interp = ((0.5 * h_max) - r_fit[0])/r_fit[1]
    f_t_interp = ((0.5 * h_max) - f_fit[0])/f_fit[1]

    fwhm = f_t_interp - r_t_interp

    return fwhm

