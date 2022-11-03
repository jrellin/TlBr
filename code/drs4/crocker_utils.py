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


def linear_interpolate_trigger2(time_bins, waveform, baseline, f=np.array([0.2]), ret_max_instead=False):
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


def histogram_FWHM(time_bin_edges, counts, outside_pts=0, given_bins=False):
    """Finds FHWM of a time histogram, need time bins and counts. Outside points refers to how many points outside
    before and after 0.5 crossings to include for linear fit"""
    from numpy.polynomial import polynomial as P
    if not given_bins:
        time_bins = 0.5 * (time_bin_edges[1:] + time_bin_edges[:-1])
    else:
        time_bins = time_bin_edges

    op = int(outside_pts)  # alias
    h_max = np.max(counts)
    h_amax = np.argmax(counts)

    lo_idx = h_amax - np.argmin(counts[:h_amax][::-1] >= (0.5 * h_max))  # just after threshold before max
    hi_idx = h_amax + np.argmin(counts[h_amax:] >= (0.5 * h_max))  # just before threshold past max

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

    # numpy.average(a, axis=None, weights=None, returned=False, *, keepdims=<no value>)
    t_center = np.average(time_bins[lo_idx:hi_idx], weights=counts[lo_idx:hi_idx])
    # counts above 50% max
    t_center_err = 1/np.sqrt(counts[lo_idx:hi_idx].sum())

    return fwhm, time_bins[h_amax], t_center, t_center_err, (time_bins[lo_idx], time_bins[hi_idx]), (r_t_interp, f_t_interp)  # fwhm, max position


class ETImage(object):
    """Used with crocker_energy_timing_det.py"""
    def __init__(self, bin_edges):
        """X axis is energy, y axis is time, range=[energy_range, time_range], bins=(energy_bins, time_bins))"""
        img_hist, self.xedges, self.yedges = np.histogram2d([], [], bins=bin_edges)
        self._img_hist = img_hist.T  # Transpose needed

    def add_values_to_image(self, energy_vals, time_vals):
        img_hist = np.histogram2d(energy_vals, time_vals, bins=[self.xedges, self.yedges])[0]
        self._img_hist += img_hist.T

    @property
    def img(self):
        return self._img_hist.copy()

    @property
    def bins(self):
        return self.xedges, self.yedges  # energy, time


def full_2D_plot(det_to_rf, det_to_t0, t0_to_rf_times, t0_to_rf_bins, detector="LFS", suppress_plots=False, method="integral"):
    """Plotting method for crocker_energy_timing_X.py files for generating 2D plots, 1D projections.
    Det_to_rf and det_to_t0 are both ETImage objects. t0_to_rf_times and t0_to_ref_bins are histogram counts,
    and bin edges, respectively. Method refers to the method to get the energy value of the detector."""
    import matplotlib.pyplot as plt

    drf_img, (drf_en_bins, drf_t_bins) = det_to_rf.img, det_to_rf.bins
    dt0_img, (dt0_en_bins, dt0_t_bins) = det_to_t0.img, det_to_t0.bins

    cbar_min, cbar_max = np.min([drf_img.min(), dt0_img.min()]), np.max([drf_img.max(), dt0_img.max()])

    fig1, (ax11, ax12, ax13) = plt.subplots(1, 3, figsize=(16, 12))  # 2 2D energy time plots, 1D energy plot
    fig2, (ax21, ax22) = plt.subplots(1, 2, figsize=(16, 12))  # 2 1D time axis projections, 1 has det_to_rf and det_to_t0

    titles = ("{d} to RF Energy-Time Histogram".format(d=detector), "{d} to t0 Energy-Time Histogram".format(d=detector))

    for ax, img, e_bins, t_bins, title in zip((ax11, ax12), (drf_img, dt0_img), (drf_en_bins, dt0_en_bins),
                                          (drf_t_bins, dt0_t_bins), titles):
        en_extent = (e_bins[0], e_bins[-1])
        t_extent = (t_bins[0], t_bins[-1])
        img_obj = ax.imshow(img, cmap='magma_r', origin='lower', interpolation='none',
                        extent=np.append(en_extent, t_extent), aspect='auto')
        ax.set_xlabel("Energy (units arb.) ", fontsize=18)
        ax.set_ylabel("$\Delta$T (ns)", fontsize=18)
        ax.set_title(title, fontsize=18)
        ax.tick_params(axis='both', labelsize=16)
        # plt.colorbar(img, fraction=0.046 * (img.shape[0] / img.shape[1]), pad=0.04)
        cbar = fig1.colorbar(img_obj, fraction=0.046, pad=0.04, ax=ax)
        img_obj.set_clim(vmin=cbar_min, vmax=cbar_max)
        cbar.draw_all()
        # if method == "integral":
        #     ax.set_xlim((0, 20))
        #     ax.set_ylim((0, 20))

    print("drf_img.shape: ", drf_img.shape)
    print("np.sum(drf_img, axis=0) shape: ", np.sum(drf_img, axis=0).shape)
    ax13.step(0.5 * (drf_en_bins[:-1] + drf_en_bins[1:]), np.sum(drf_img, axis=0), '-', where='mid')  # either drf_img or dt0_img could be used
    ax13.set_xlabel("Energy (units arb.)", fontsize=18)
    ax13.set_ylabel("Counts", fontsize=18)
    ax13.tick_params(axis='both', labelsize=16)
    # if method == "integral":
    #    ax13.set_xlim((0, 20))
    # bins = (bin_edges[1:] + bin_edges[:-1]) / 2
    #                 ax.step(bins, values, 'b-', where='mid', label=plot_label)
    #                 ax.set_xlabel(xlbl, fontsize=18)
    #                 ax.set_ylabel("counts", fontsize=18)
    #                 ax.set_title(title, fontsize=18)
    #                 ax.tick_params(axis='both', labelsize=16)

    fig1.canvas.draw()
    fig1.canvas.flush_events()

    for ax, bin_edges, values, \
        xlbl, title, plot_label, lcolor in zip((ax21, ax22), (t0_to_rf_bins, drf_t_bins),
                                             (t0_to_rf_times, np.sum(drf_img, axis=1)),
                                             ("$\Delta$T (ns)", "$\Delta$T (ns)",),
                                             ("$T_{t0}-T_{rf}$", "$\Delta$T with RF or T0"),
                                             ("t0 to RF", "$T_{\gamma}-T_{rf}$"),
                                             ('k-', 'b-')):
        bins = (bin_edges[1:] + bin_edges[:-1]) / 2
        ax.step(bins, values, lcolor, where='mid', label=plot_label)
        ax.set_xlabel(xlbl, fontsize=18)
        ax.set_ylabel("counts", fontsize=18)
        ax.set_title(title, fontsize=18)
        ax.tick_params(axis='both', labelsize=16)

    ax22.step(0.5 * (dt0_t_bins[1:] + dt0_t_bins[:-1]), np.sum(dt0_img, axis=1), 'g-', where='mid', label="$T_{\gamma}-T_{t0}$")
    ax22.legend(loc='best')

    fig1.tight_layout()
    fig2.tight_layout()
    if not suppress_plots:
        plt.show()


def global_limits(imgs):
    """imgs is a list of 2D imgs. Not used"""
    min_val = np.inf
    max_val = 0
    for img in imgs:
        if img.min() < min_val:
            min_val = img.min()
        if img.max() > max_val:
            max_val = img.max()
    return min_val, max_val


    # x_extent = (en_edges[0], en_edges[-1])
    #  = (t_edges[0] * self.sample_period, t_edges[-1] * self.sample_period)

def plot_lfs_shifts():
    import os
    from pathlib import Path
    import matplotlib.pyplot as plt

    positions = [0, 2, 4, 5, 6, 8]

    data_list = ["20221017_Crocker_31.6V_LFS_500pa_SingleDataset_nim_amp_p0_v20",
                 "20221017_Crocker_31.6V_LFS_500pa_SingleDataset_nim_amp_p2_v19",
                 "20221017_Crocker_31.6V_LFS_500pa_SingleDataset_nim_amp_p4_v18",
                 "20221017_Crocker_31.6V_LFS_500pa_DualDataset_nim_amp_p5_v17",
                 "20221017_Crocker_31.6V_LFS_500pa_DualDataset_nim_amp_p6_v16",
                 "20221017_Crocker_31.6V_LFS_500pa_DualDataset_nim_amp_p8_v15"]

    data_list2 = ["20221017_Crocker_31.6V_LFS_500pa_SingleDataset_nim_amp_p0_v20_en_gated",
                  "20221017_Crocker_31.6V_LFS_500pa_SingleDataset_nim_amp_p2_v19_en_gated",
                  "20221017_Crocker_31.6V_LFS_500pa_SingleDataset_nim_amp_p4_v18_en_gated",
                  "20221017_Crocker_31.6V_LFS_500pa_DualDataset_nim_amp_p5_v17_en_gated",
                  "20221017_Crocker_31.6V_LFS_500pa_DualDataset_nim_amp_p6_v16_en_gated",
                  "20221017_Crocker_31.6V_LFS_500pa_DualDataset_nim_amp_p8_v15_en_gated"]

    data_cherenkov = ["20221017_Crocker_31.6V_cherenkov_500pa_DualDataset_nim_amp_p0_v9",
                      "20221017_Crocker_31.6V_cherenkov_500pa_DualDataset_nim_amp_p2_v10",
                      "20221017_Crocker_31.6V_cherenkov_500pa_DualDataset_nim_amp_p4_v11",
                      "20221017_Crocker_31.6V_cherenkov_500pa_DualDataset_nim_amp_p5_v14",
                      "20221017_Crocker_31.6V_cherenkov_500pa_DualDataset_nim_amp_p6_v12",
                      "20221017_Crocker_31.6V_cherenkov_500pa_DualDataset_nim_amp_p8_v13"]

    data_cherenkov_gated = ["20221017_Crocker_31.6V_cherenkov_500pa_DualDataset_nim_amp_p0_v9_en_gated",
                            "20221017_Crocker_31.6V_cherenkov_500pa_DualDataset_nim_amp_p2_v10_en_gated",
                            "20221017_Crocker_31.6V_cherenkov_500pa_DualDataset_nim_amp_p4_v11_en_gated",
                            "20221017_Crocker_31.6V_cherenkov_500pa_DualDataset_nim_amp_p5_v14_en_gated",
                            "20221017_Crocker_31.6V_cherenkov_500pa_DualDataset_nim_amp_p6_v12_en_gated",
                            "20221017_Crocker_31.6V_cherenkov_500pa_DualDataset_nim_amp_p8_v13_en_gated"]

    # np.savez(save_fname, filename=self.filename,
    #                          t0_to_rf_bins=t0_to_rf_bins, t0_to_rf_counts=t0_to_rf_times,
    #                          det_to_ref_time_bins=t_bins,
    #                          det_to_rf_counts=det_to_rf_times, det_to_t0_counts=det_to_t0_times)

    det_to_rf = {"max time": [], "FWHMs": [], "time centers": [], "time_centers_err": []}
    det_to_t0 = {"max time": [], "FWHMs": [], "time centers": [], "time_centers_err": []}

    outside_pts = 0

    fig2, axs = plt.subplots(1, 1, figsize=(16, 12))  # axis shift
    fig3, axrf = plt.subplots(1, 1, figsize=(16, 12))  # axis shift
    # ax.step(bins, counts / counts.sum(), where='mid', label=label)

    # for file in data_list, data_list2, data_cherenkov, data_cherenkov_gated
    for file, pos in zip(data_cherenkov, positions):
        # TODO: remove
        if pos not in (0, 4, 8):
            continue
        fname = os.path.join(str(Path(os.getcwd()).parents[1]), "sample_data", "drs4", file + ".npz")
        # fname = file + ".npz"
        data = np.load(fname)

        drf_fwhm, drf_max, trf_c, trf_c_err, (t_lo_rf, t_hi_rf), interps = histogram_FWHM(data['det_to_ref_time_bins'], data['det_to_rf_counts'], outside_pts=outside_pts)
        dt0_fwhm, dt0_max, tt0_c, tt0_c_err, (t_lo_t0, t_hi_t0), interps = histogram_FWHM(data['det_to_ref_time_bins'], data['det_to_t0_counts'], outside_pts=outside_pts)

        if pos == 0:
            print("Position 0 cross points:")
            print("RF (ns): ", (t_lo_rf, t_hi_rf))
            print("RF half max: ", data['det_to_rf_counts'].max()/2)
            print("T0 (ns): ", (t_lo_t0, t_hi_t0))
            print("T0 half max: ", data['det_to_t0_counts'].max() / 2)

        det_to_rf['max time'].append(drf_max)
        det_to_rf['FWHMs'].append(drf_fwhm)
        det_to_rf['time centers'].append(trf_c)
        det_to_rf['time_centers_err'].append(trf_c_err)

        det_to_t0['max time'].append(dt0_max)
        det_to_t0['FWHMs'].append(dt0_fwhm)
        det_to_t0['time centers'].append(tt0_c)
        det_to_t0['time_centers_err'].append(tt0_c_err)

        bin_edges = data['det_to_ref_time_bins']
        trf_bin_edges = data['t0_to_rf_bins']
        # axs.plot(0.5 * (bin_edges[1:] + bin_edges[:-1]), data['det_to_t0_counts']/data['det_to_t0_counts'].max(), label=pos)
        # axrf.plot(0.5 * (trf_bin_edges[1:] + trf_bin_edges[:-1]), data['t0_to_rf_counts']/data['t0_to_rf_counts'].max(), label=pos)
        axs.plot(0.5 * (bin_edges[1:] + bin_edges[:-1]), data['det_to_t0_counts'], label=pos)
        axrf.plot(0.5 * (trf_bin_edges[1:] + trf_bin_edges[:-1]), data['t0_to_rf_counts'], label=pos)

    axs.legend(loc='best')
    axs.set_xlabel('ns')
    axs.set_ylabel('counts')

    axrf.legend(loc='best')
    axrf.set_xlabel('ns')
    axs.set_ylabel('counts')

    print("Statistics:")
    print("det to RF: ", det_to_rf)
    print("det to t0: ", det_to_t0)
    print("positions: ", positions)
    # histogram_FWHM(time_bin_edges, counts, outside_pts=0)

    # fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 12))

    # det = "LFS"
    # det = "Cher."
    # ax1.plot(np.array(positions), det_to_rf['time centers'], label=det + ' to RF')
    # ax1.plot(np.array(positions), det_to_t0['time centers'], label=det + ' to T0')
    # ax1.set_xlabel('relative pos (cm)')
    # ax1.set_ylabel('center of distribution (ns)')
    # ax1.legend(loc='best')

    # ax2.plot(np.array(positions), det_to_rf['FWHMs'], label=det + ' to RF')
    # ax2.plot(np.array(positions), det_to_t0['FWHMs'], label=det + ' to T0')
    # ax2.set_xlabel('relative pos (cm)')
    # ax2.set_ylabel('FWHM (ns)')
    # ax2.legend(loc='best')
    plt.show()


def plot_lfs_shifts_averaging():
    import os
    from pathlib import Path
    import matplotlib.pyplot as plt

    # positions = [0, 2, 4, 5, 6, 8]
    trials = [1, 2, 3, 4, 5, 6]

    data_cherenkov = ["20221017_Crocker_31.6V_cherenkov_500pa_DualDataset_nim_amp_p0_v9",
                      "20221017_Crocker_31.6V_cherenkov_500pa_DualDataset_nim_amp_p2_v10",
                      "20221017_Crocker_31.6V_cherenkov_500pa_DualDataset_nim_amp_p4_v11",
                      "20221017_Crocker_31.6V_cherenkov_500pa_DualDataset_nim_amp_p5_v14",
                      "20221017_Crocker_31.6V_cherenkov_500pa_DualDataset_nim_amp_p6_v12",
                      "20221017_Crocker_31.6V_cherenkov_500pa_DualDataset_nim_amp_p8_v13"]

    det_to_rf = {"max time": [], "FWHMs": [], "time centers": [], "time_centers_err": []}
    det_to_t0 = {"max time": [], "FWHMs": [], "time centers": [], "time_centers_err": []}

    outside_pts = 0

    fig1, (axrf, axc) = plt.subplots(1, 2, figsize=(16, 12))  # t0 to RF, averaged det to T0 and det to RF curves

    # initialize with first entry
    fname = os.path.join(str(Path(os.getcwd()).parents[1]), "sample_data", "drs4", data_cherenkov[0] + ".npz")
    data = np.load(fname)
    time_bin_edges = data['det_to_ref_time_bins']
    t0_to_rf_bin_edges = data['t0_to_rf_bins']
    counts_rf = data['det_to_rf_counts']
    counts_t0 = data['det_to_t0_counts']

    drf_fwhm, drf_max, trf_c, trf_c_err, (t_lo_rf, t_hi_rf), interps = histogram_FWHM(time_bin_edges, counts_rf, outside_pts=outside_pts)
    dt0_fwhm, dt0_max, tt0_c, tt0_c_err, (t_lo_t0, t_hi_t0), interps = histogram_FWHM(time_bin_edges, counts_t0, outside_pts=outside_pts)
    det_to_rf['max time'].append(drf_max)
    det_to_rf['FWHMs'].append(drf_fwhm)
    det_to_rf['time centers'].append(trf_c)
    det_to_rf['time_centers_err'].append(trf_c_err)

    det_to_t0['max time'].append(dt0_max)
    det_to_t0['FWHMs'].append(dt0_fwhm)
    det_to_t0['time centers'].append(tt0_c)
    det_to_t0['time_centers_err'].append(tt0_c_err)

    total_rf = counts_rf.copy()
    total_t0 = counts_t0.copy()

    shift = 0
    det_time_bins = 0.5 * (time_bin_edges[1:] + time_bin_edges[:-1])
    t0_to_rf_bins = 0.5 * (t0_to_rf_bin_edges[1:] + t0_to_rf_bin_edges[:-1]) + shift

    axrf.plot(t0_to_rf_bins, data['t0_to_rf_counts'], label='Trial ' + str(trials[0]))

    for file, trial in zip(data_cherenkov[1:], trials[1:]):
        fname = os.path.join(str(Path(os.getcwd()).parents[1]), "sample_data", "drs4", file + ".npz")
        # fname = file + ".npz"
        data = np.load(fname)
        total_rf += data['det_to_rf_counts']
        total_t0 += data['det_to_t0_counts']
        # print("Total RF counts: ", total_rf.sum())
        # print("Total t0 counts: ", total_t0.sum())

        drf_fwhm, drf_max, trf_c, trf_c_err, (t_lo_rf, t_hi_rf), interps = histogram_FWHM(data['det_to_ref_time_bins'],
                                                                                 data['det_to_rf_counts'],
                                                                                 outside_pts=outside_pts)
        dt0_fwhm, dt0_max, tt0_c, tt0_c_err, (t_lo_t0, t_hi_t0), interps = histogram_FWHM(data['det_to_ref_time_bins'],
                                                                                 data['det_to_t0_counts'],
                                                                                 outside_pts=outside_pts)
        det_to_rf['max time'].append(drf_max)
        det_to_rf['FWHMs'].append(drf_fwhm)
        det_to_rf['time centers'].append(trf_c)
        det_to_rf['time_centers_err'].append(trf_c_err)

        det_to_t0['max time'].append(dt0_max)
        det_to_t0['FWHMs'].append(dt0_fwhm)
        det_to_t0['time centers'].append(tt0_c)
        det_to_t0['time_centers_err'].append(tt0_c_err)

        # axrf.step(t0_to_rf_bins, data['t0_to_rf_counts'] / data['t0_to_rf_counts'].max(), where='mid', label=trial)
        axrf.step(t0_to_rf_bins, data['t0_to_rf_counts'], where='mid', label='Trial ' + str(trial))

    axc.step(det_time_bins, total_rf, 'tab:purple', where='mid', label="$T_{\gamma}-T_{rf}$")
    axc.step(det_time_bins, total_t0, 'tab:cyan', where='mid', label="$T_{\gamma}-T_{T0}$")

    axc.legend(loc='best', fontsize=18)
    axc.set_xlabel("$\Delta$T (ns)", fontsize=18)
    axc.set_ylabel("Counts", fontsize=18)
    axc.tick_params(axis='both', labelsize=16)
    axc.set_title("$T_{t0}-T_{rf}$", fontsize=18)
    axc.set_title("$\Delta$T with RF or T0", fontsize=22)
    axc.set_xlim((0, 10))

    axrf.legend(loc='best', fontsize=18)
    axrf.set_xlabel("$\Delta$T (ns)", fontsize=18)  # "$\Delta$T with RF or T0"
    axrf.set_ylabel('Counts', fontsize=18)
    axrf.tick_params(axis='both', labelsize=16)
    axrf.set_title("$T_{T0}-T_{rf}$", fontsize=22)
    axrf.set_xlim((-6.5, 1))

    for counts, color, left_shift, right_shift, lbl in zip((total_rf, total_t0), ('tab:purple', 'tab:cyan'), (1, 2), (1, 0),
                                                           ('rf', 't0')):
        (h_max, h_amax), (lo_idx, hi_idx) = _histogram_t0_stats(counts)
        # return (h_max, h_amax), (lo_idx, hi_idx)  # fwhm, max position
        fwhm, _, _, c_err, _, interps = histogram_FWHM(det_time_bins, counts, outside_pts=outside_pts)
        print('FWHM {l}: '.format(l=lbl), fwhm)
        print("Err: ", c_err)
        print("Weighted Average Uncertainty: ", 1 / np.sqrt(counts[lo_idx:hi_idx].sum()))
        # axc.hlines(h_max / 2, det_time_bins[lo_idx-left_shift], det_time_bins[hi_idx+right_shift], colors=color, linestyles='dashed', linewidths=4)
        axc.hlines(h_max / 2, interps[0], interps[1], colors=color,
                   linestyles='dashed', linewidths=4)

    # print("Statistics:")
    # print("det to RF: ", det_to_rf)
    # print("det to t0: ", det_to_t0)
    # histogram_FWHM(time_bin_edges, counts, outside_pts=0)

    plt.show()

def save_current_t0_max_arrays():
    # TODO: Modify these plots
    import os
    from pathlib import Path
    import matplotlib.pyplot as plt
    import crocker_energy_timing_lfs as cet

    lfs_data_file_name = "20221017_Crocker_31.6V_LFS_500pa_SingleDataset_nim_amp_p2_v19.dat"  # p2 LFS
    lfs_det_name = "lfs_en"  # no det field for this file
    lfs_fname = os.path.join(str(Path(os.getcwd()).parents[1]), "sample_data", "drs4", lfs_data_file_name)
    cet.t0_heights(lfs_fname, det_type=lfs_det_name, output=True, suppress_plots=True)

    cher_data_file_name = "20221017_Crocker_31.6V_cherenkov_400pa_DualDataset_v4.dat"
    cher_det_name = "cherenkov"
    cher_fname = os.path.join(str(Path(os.getcwd()).parents[1]), "sample_data", "drs4", cher_data_file_name)
    cet.t0_heights(cher_fname, det_type=cher_det_name, output=True, suppress_plots=False)


def _histogram_t0_stats(counts):
    """Stats useful for plotting on t0 plot"""

    h_max = np.max(counts)
    h_amax = np.argmax(counts)

    lo_idx = h_amax - np.argmin(counts[:h_amax][::-1] >= (0.5 * h_max))  # just after threshold before max
    hi_idx = h_amax + np.argmin(counts[h_amax:] >= (0.5 * h_max))  # just before threshold past max

    return (h_max, h_amax), (lo_idx, hi_idx)  # fwhm, max position

def current_vs_t0voltage_plots():
    # TODO: Figure out why can't directly output t0_heights
    import os
    from pathlib import Path
    import matplotlib.pyplot as plt

    base = "C:/Users/justi/Documents/GitHub/TlBr/sample_data/drs4/"
    f4 = "20221017_Crocker_31.6V_cherenkov_400pa_DualDataset_v4t0max.npz"  # 400 pA
    f5 = "20221017_Crocker_31.6V_LFS_500pa_SingleDataset_nim_amp_p2_v19t0max.npz"  # 500 pA

    cvals = np.load(base + f4)
    lvals = np.load(base + f5)

    fig, ax = plt.subplots(1, 1, figsize=(16, 12))  # ax1 -> rise_times, ax2 -> amplitudes
    fig.suptitle("T0 Max Pulse Voltage (Normalized)", fontsize=22)
    bins = lvals['bins']  # should be the same for both
    # for counts, label in zip((cvals['counts'], lvals['counts']), ("400 pA", "500 pA")):
    for counts, label, color in zip(((lvals['counts']), (cvals['counts'])), ("400 pA", "500 pA"), ('b', 'g')):
        # if label == "400 pA":
        #     continue
        norm_counts = counts/counts.sum()
        ax.step(bins, norm_counts, color, where='mid', label=label)
        (h_max, h_amax), (lo_idx, hi_idx) = _histogram_t0_stats(norm_counts)
        wa = np.average(bins, weights=counts)
        wa_h = norm_counts[h_amax + (np.abs(bins[h_amax:] - wa)).argmin()]

        # ax.vlines(wa, 0, wa_h, colors=color, linestyles='dashed', linewidths=4)
        # ax.hlines(h_max/2, bins[lo_idx-12], bins[hi_idx+12], colors=color, linestyles='solid', linewidths=4)
        print("Current {c} Weighted Average: ".format(c=label), wa)
        print("Weighted Average Uncertainty: ", 1/np.sqrt(counts[lo_idx:hi_idx].sum()))
        print("Total counts: ", counts.sum())
        print("FWHM: ", bins[hi_idx] - bins[lo_idx])
    ax.set_xlabel("Max Voltage (V)", fontsize=18)
    ax.set_ylabel("counts", fontsize=18)
    ax.tick_params(axis='both', labelsize=16)
    ax.legend(loc='best', fontsize=20)
    plt.show()


if __name__ == "__main__":
    # plot_lfs_shifts()
    # plot_lfs_shifts_averaging()
    current_vs_t0voltage_plots()  # t0 max voltage vs current
    # save_current_t0_max_arrays()
    # TODO: histogram_t0_FWHM for mean
