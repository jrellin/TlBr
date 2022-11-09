import numpy as np
import tables
from dds_utils import load_h5file
import matplotlib.pyplot as plt


def anode_to_cherenkov_plot_crocker(fname):
    pin_map = np.array([[2, 1, 3, 4], [15, 8, 6, 5], [13, 11, 9, 7], [16, 14, 12, 10]])
    # channel map as pins as plugged into digitizer
    det_map = np.arange(1, 16 + 1).reshape([4, 4])

    h5file = load_h5file(fname)  # h5 file obj
    print("Print h5file:")
    print(h5file)

    cci_table = h5file.root.cci_event_data
    sipm_table = h5file.root.sipm_event_data

    nevts = cci_table.nrows

    anode_evts = np.zeros([nevts, 16])
    for i in np.arange(16):
        anode_evts[:, i] = cci_table.col('amp' + str(i+1)) - cci_table.col('bl'+str(i+1))
    max_anode = np.max(anode_evts, axis=1)
    print("Max_anode.size: ", max_anode.size)

    # integral
    cher_en_int_bins = np.linspace(0, 30, num=4097)

    ac2d_hist, a_edges, c_edges = np.histogram2d(max_anode, sipm_table.col('det_en_integral'), bins=[(2**12)//4, cher_en_int_bins])
    ac2d_hist = ac2d_hist.T

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.imshow(ac2d_hist, cmap='magma_r', origin='lower', interpolation='none',
              extent=[a_edges[0], a_edges[-1], c_edges[0], c_edges[-1]],
              aspect='auto')
    ax.set_xlabel("Max ADC Bin")
    ax.set_ylabel("SIPM (Cherenkov) Energy Integral")

    plt.show()

    fig1, ax1 = plt.subplots(1, 1, figsize=(8, 6))
    ax1.scatter(max_anode, sipm_table.col('det_en_integral'), s=0.5)
    ax1.set_xlabel("Max ADC Bin")
    ax1.set_ylabel("SIPM (Cherenkov) Energy Integral")

    plt.show()

    h5file.close()


def anode_to_cherenkov_plot_GBSF(fname):
    pin_map = np.array([[2, 1, 3, 4], [15, 8, 6, 5], [13, 11, 9, 7], [16, 14, 12, 10]])
    # channel map as pins as plugged into digitizer
    det_map = np.arange(1, 16 + 1).reshape([4, 4])

    h5file = load_h5file(fname)  # h5 file obj
    print("Print h5file:")
    print(h5file)

    cci_table = h5file.root.cci_event_data
    sipm_table = h5file.root.sipm_event_data

    nevts = cci_table.nrows

    anode_evts = np.zeros([nevts, 16])
    overflow_check = np.zeros([nevts, 16])

    for i in np.arange(16):
        anode_evts[:, i] = cci_table.col('amp' + str(i+1)) - cci_table.col('bl'+str(i+1))
        overflow_check[:, i] = cci_table.col('amp' + str(i + 1))

    max_anode = np.max(anode_evts, axis=1)
    overflow = (np.max(overflow_check, axis=1) >= 4095)

    print("Max_anode.size: ", max_anode.size)
    print("Overflow events: ", overflow.sum())

    overflow_evts = 0
    used_evts = 0

    max_anode_no_overflow = np.zeros(nevts)
    cher_int_no_overflow = np.zeros(nevts)

    integral = False  # otherwise peak

    for n in np.arange(nevts):
        if overflow[n]:  # filter overflow events
            overflow_evts += 1
            continue
        sipm_evt = sipm_table.read(n, n + 1)
        # det_trig = sipm_evt["det_trig"]
        if integral:
            en = sipm_evt["det_en_integral"]
        else:
            en = sipm_evt["det_en_peak"]

        max_anode_no_overflow[used_evts] = max_anode[n]
        cher_int_no_overflow[used_evts] = en

        used_evts += 1

    print("Total Events: ", nevts)
    print("Used Events: ", used_evts)
    print("Overflow Events: ", overflow_evts)

    cut_max_anode = max_anode_no_overflow[:used_evts]
    cut_cher_int = cher_int_no_overflow[:used_evts]

    # peak, en_bins = np.linspace(0, 0.7, num=4097), en_low_gate = 0.08
    # integral, en_bins = np.linspace(0, 30, num=4097), en_log_gate = 3.5
    # cher_en_int_bins = np.linspace(0, 30, num=2049)  # 4097, 2049, 1025
    if integral:
        cher_en_int_bins = np.linspace(0, 10, num=1025)
    else:  # peak
        cher_en_int_bins = np.linspace(0, 0.7, num=1025)
    # 4097, 2049, 1025

    ac2d_hist, a_edges, c_edges = np.histogram2d(cut_max_anode, cut_cher_int, bins=[(2**12)//4, cher_en_int_bins])
    ac2d_hist = ac2d_hist.T

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ac2d_hist_filt = ac2d_hist.copy()
    ac2d_hist_filt[:, :40] = 0
    ax.imshow(ac2d_hist_filt, cmap='magma_r', origin='lower', interpolation='none',
              extent=[a_edges[0], a_edges[-1], c_edges[0], c_edges[-1]],
              aspect='auto')
    ax.set_xlabel("Max ADC Bin", fontsize=18)
    ax.set_ylabel("SIPM (Cherenkov) Energy Integral", fontsize=18)
    ax.tick_params(axis='both', labelsize=16)

    plt.show()

    fig1, ax1 = plt.subplots(1, 1, figsize=(8, 6))
    ax1.scatter(cut_max_anode, cut_cher_int, s=0.2)
    ax1.set_xlabel("Max ADC Bin", fontsize=18)
    ax1.set_ylabel("SIPM (Cherenkov) Energy Integral", fontsize=18)
    ax1.tick_params(axis='both', labelsize=16)

    plt.show()

    h5file.close()


def anode_to_t0_rf_crocker(fname, suppress_plots=False):
    pin_map = np.array([[2, 1, 3, 4], [15, 8, 6, 5], [13, 11, 9, 7], [16, 14, 12, 10]])
    # channel map as pins as plugged into digitizer
    det_map = np.arange(1, 16 + 1).reshape([4, 4])

    h5file = load_h5file(fname)  # h5 file obj
    print("Print h5file:")
    print(h5file)

    cci_table = h5file.root.cci_event_data
    sipm_table = h5file.root.sipm_event_data

    nevts = cci_table.nrows

    filter_str = '(amp1 < 4095)'  # start
    for n in np.arange(2, 16 + 1):
        filter_str += '& (amp' + str(n) + '< 4095) '
    filter_str = filter_str[:-1]  # remove that last space

    anode_evts = np.zeros([nevts, 16])
    overflow_check = np.zeros([nevts, 16])

    for i in np.arange(16):
        anode_evts[:, i] = cci_table.col('amp' + str(i + 1)) - cci_table.col('bl' + str(i + 1))
        overflow_check[:, i] = cci_table.col('amp' + str(i + 1))

    overflow = (np.max(overflow_check, axis=1) >= 4095)
    max_anode = np.max(anode_evts, axis=1)

    print("Max_anode.size: ", max_anode.size)
    print("Overflow events: ", overflow.sum())

    delay = {"rf": 0, "cherenkov": 0.7, "t0": 12.81}  # Cable delays, original
    # delay = {"rf": 0, "cherenkov": 0.7, "t0": 8}  # Cable delays, 8 could work

    max_anode_w_triggers = np.zeros(nevts)
    delta_rf_triggers = np.zeros(nevts)  # delta RF -> cherenkov trigger - RF
    delta_t0_triggers = np.zeros(nevts)

    # {"drs4_evt_id": self.event.event_id,
    #                     "rf_zero_cs": rfc, "rf_zero_cs_signs": rfcs,
    #                     "t0_trigs": t0_trigs, "t0_v_at_trig": t0_voltage_at_trig,
    #                     "det_trig": det_trig, "det_v_at_trig": det_voltage_at_trig,
    #                     "det_en_peak": cher_en_peak, "det_en_integral": cher_en_integral}

    missed_evts = 0
    used_evts = 0

    for n in np.arange(nevts):
        if overflow[n]:  # filter overflow events
            continue
        sipm_evt = sipm_table.read(n, n + 1)
        t0_trgs = sipm_evt["t0_trigs"] - delay['t0']
        det_trig = sipm_evt["det_trig"] - delay['cherenkov']
        rf_zeros = sipm_evt["rf_zero_cs"] - delay['rf']

        rf_ref_idx = ((det_trig - rf_zeros) > 0)
        t0_ref_idx = ((det_trig - t0_trgs) > 0)

        if (rf_ref_idx.sum() <= 0) or (t0_ref_idx.sum() <= 0):
            missed_evts += 1  # either no RF or T0 reference
            continue

        tgamma_to_rf = (det_trig - rf_zeros)[rf_ref_idx].min()
        tgamma_to_t0 = (det_trig - t0_trgs)[t0_ref_idx].min()

        max_anode_w_triggers[used_evts] = max_anode[n]
        delta_rf_triggers[used_evts] = tgamma_to_rf
        delta_t0_triggers[used_evts] = tgamma_to_t0

        used_evts += 1

    print("Total Events: ", nevts)
    print("Used Events: ", used_evts)
    print("Missed Events: ", missed_evts)
    cut_max_anode = max_anode_w_triggers[:used_evts]
    cut_drf = delta_rf_triggers[:used_evts]
    cut_t0 = delta_t0_triggers[:used_evts]

    if not suppress_plots:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 12))
        ax1.scatter(cut_drf, cut_max_anode, s=0.5)
        ax1.set_xlabel("$\Delta$T (ns)", fontsize=18)
        ax1.set_ylabel("Max ADC Bin", fontsize=18)
        ax1.set_title("$T_{\gamma}-T_{rf}$ vs. Charge Induction (Max Anode)", fontsize=18)
        ax1.set_xlim(-2, 22)
        ax1.tick_params(axis='both', labelsize=16)

        ax2.scatter(cut_t0, cut_max_anode, s=0.5)
        ax2.set_xlabel("$\Delta$T (ns)", fontsize=18)
        ax2.set_ylabel("Max ADC Bin", fontsize=18)
        ax2.set_title("$T_{\gamma}-T_{T0}$ vs. Charge Induction (Max Anode)", fontsize=18)
        ax2.set_xlim(-2, 22)
        ax2.tick_params(axis='both', labelsize=16)

        fig.tight_layout()

        plt.show()

    e_range = [0, 4096]
    e_bins = 4096//4
    del_t_range = [-2, 22]
    del_t_bins = 481
    Hdrf, rf_xe, rf_ye = np.histogram2d(cut_drf, cut_max_anode, bins=[del_t_bins, e_bins], range=[del_t_range, e_range])
    Hdt0, t0_xe, t0_ye = np.histogram2d(cut_t0, cut_max_anode, bins=[del_t_bins, e_bins], range=[del_t_range, e_range])

    if suppress_plots:
        return (Hdrf.T, rf_xe, rf_ye), (Hdt0.T, t0_xe, t0_ye)

    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 12))

    ax1.imshow(Hdrf.T, cmap='magma_r', origin='lower', interpolation='none',
               extent=[rf_xe[0], rf_xe[-1], rf_ye[0], rf_ye[-1]],
               aspect='auto')
    ax1.set_xlabel("$\Delta$T (ns)", fontsize=18)
    ax1.set_ylabel("Max ADC Bin", fontsize=18)
    ax1.set_title("$T_{\gamma}-T_{rf}$ vs. Charge Induction (Max Anode)", fontsize=18)
    ax1.tick_params(axis='both', labelsize=16)

    ax2.imshow(Hdt0.T, cmap='magma_r', origin='lower', interpolation='none',
               extent=[t0_xe[0], t0_xe[-1], t0_ye[0], t0_ye[-1]],
               aspect='auto')
    ax2.set_xlabel("$\Delta$T (ns)", fontsize=18)
    ax2.set_ylabel("Max ADC Bin", fontsize=18)
    ax2.set_title("$T_{\gamma}-T_{T0}$ vs. Charge Induction (Max Anode)", fontsize=18)
    ax2.tick_params(axis='both', labelsize=16)

    fig2.tight_layout()

    plt.show()

    h5file.close()

    return (Hdrf.T, rf_xe, rf_ye), (Hdt0.T, t0_xe, t0_ye)


def stage_plots(cor_base, cor_files, pos=(0, 8)):

    fig1, (ax11, ax12) = plt.subplots(1, 2, figsize=(16, 12))
    fig2, (ax21, ax22) = plt.subplots(1, 2, figsize=(16, 12))

    for pos, file in zip(pos, cor_files):
        (hdrf, rf_xe, rf_ye), (hdt0, t0_xe, t0_ye) = anode_to_t0_rf_crocker(cor_base + file, suppress_plots=True)
        ax11.step(0.5 * (rf_xe[1:] + rf_xe[:-1]), hdrf.sum(axis=0), where='mid', label=str(pos) + ' mm')
        ax12.step(0.5 * (rf_ye[1:] + rf_ye[:-1]), hdrf.sum(axis=1), where='mid', label=str(pos) + ' mm')
        ax21.step(0.5 * (t0_xe[1:] + t0_xe[:-1]), hdt0.sum(axis=0), where='mid', label=str(pos) + ' mm')
        ax22.step(0.5 * (t0_ye[1:] + t0_ye[:-1]), hdt0.sum(axis=1), where='mid', label=str(pos) + ' mm')
    ax11.legend(loc='best')
    ax12.legend(loc='best')
    ax21.legend(loc='best')
    ax22.legend(loc='best')
    plt.show()


def main():
    import os
    from pathlib import Path
    cor_base = "C:/Users/justi/Documents/GitHub/TlBr/code/dual_data_sets/Data_correlated/"
    cor_files = ["DavisD2022_10_17T13_57_clean_Crocker_combined.h5", "DavisD2022_10_17T14_10_clean_Crocker_combined.h5",
                 "DavisD2022_10_17T14_25_clean_Crocker_combined.h5", "DavisD2022_10_17T15_6_clean_Crocker_combined.h5",
                 "DavisD2022_10_17T14_39_clean_Crocker_combined.h5", "DavisD2022_10_17T14_52_clean_Crocker_combined.h5"]
    positions = [0, 2,
                 4, 5,
                 6, 8]  # 0, 1, 2, 3, 4, 5
    files_to_include = [1, 2, 4]
    cor_files_include = [cor_files[i] for i in files_to_include]
    pos_include = [positions[i] for i in files_to_include]

    stage_plots(cor_base, cor_files_include, pos=pos_include)


def main_single_plots():
    cor_base = "C:/Users/justi/Documents/GitHub/TlBr/code/dual_data_sets/Data_correlated/"
    cor_files = ["DavisD2022_10_17T13_57_clean_Crocker_combined.h5", "DavisD2022_10_17T14_10_clean_Crocker_combined.h5",
                 "DavisD2022_10_17T14_25_clean_Crocker_combined.h5", "DavisD2022_10_17T15_6_clean_Crocker_combined.h5",
                 "DavisD2022_10_17T14_39_clean_Crocker_combined.h5", "DavisD2022_10_17T14_52_clean_Crocker_combined.h5"]
    anode_to_t0_rf_crocker(cor_base + cor_files[0])


def main_button_source():
    cor_base = "C:/Users/justi/Documents/GitHub/TlBr/code/dual_data_sets/Data_correlated/"
    # cor_file = "DavisD2022_11_1T11_37_clean_GBSF_combined.h5"  # Na-22, set 1
    cor_file = "DavisD2022_11_3T14_52_clean_GBSF_combined.h5"  # Na-22, MUCH higher threshold

    anode_to_cherenkov_plot_GBSF(cor_base + cor_file)


if __name__ == "__main__":
    # main()
    # main_single_plots()
    main_button_source()
