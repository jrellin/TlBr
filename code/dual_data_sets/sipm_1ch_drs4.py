from binio import DRS4BinaryFile
import numpy as np
import matplotlib.pyplot as plt
from dds_utils import *
# copy-pasted crocker_energy_timing_lfs.py
# working on: 10/31


class SiPM1(object):

    def __init__(self, filename):  # cherenkov or lfs_en
        self.filename = filename
        self.f = DRS4BinaryFile(filename)
        # aliases
        self.n_boards = self.f.num_boards
        self.board_ids = self.f.board_ids
        self.time_widths = self.f.time_widths
        self.channels = self.f.channels
        self.n_channels = [len(self.channels[b]) for b in self.board_ids]
        self.event = next(self.f)  # first event
        self.event_number = 1
        self.ch_time_bins = np.zeros(1024)  # temporary working memory for time calibration data
        self.buffer = np.zeros(1024)  # temporary working memory for voltage calibration data
        # self.ch_names = ["rf", "lfs", "cherenkov", "t0"]
        # 12.81
        # self.cable_delays = {1: 0}  # No delays with 1 channel
        self.det_type = "cherenkov"  # really trigger channel
        self.ch_names = {"cherenkov": 1}
        self.ch_numbers = {1: "cherenkov"}

    def event_voltage_calibrate(self, board, chns, verbose=False):
        voltage_calibrated = {}
        for chn in chns:
            voltage_calibrated[chn] = (self.event.adc_data[board][chn] / 65536) + \
                                      (self.event.range_center / 1000) - 0.5
        return voltage_calibrated

    def event_timing_calibrate(self, board, chns, ref=1):  # check ref is in chns before this gets called
        channels = np.array(chns)
        time_calibrated = {}

        trg_cell = self.event.trigger_cells[board]
        self.ch_time_bins[:] = np.cumsum(np.concatenate((np.array([0]), self.time_widths[board][ref][trg_cell:1023],
                                                         self.time_widths[board][ref][:trg_cell])))
        time_calibrated[ref] = self.ch_time_bins.copy()
        ref_ch_0cell = self.ch_time_bins[(1024 - trg_cell) % 1024]

        for chn in channels[channels != ref]:
            self.ch_time_bins[:] = np.cumsum(np.concatenate((np.array([0]), self.time_widths[board][chn][trg_cell:1023],
                                                             self.time_widths[board][chn][:trg_cell])))
            ch_0cell = self.ch_time_bins[(1024 - trg_cell) % 1024]
            self.ch_time_bins += (ref_ch_0cell - ch_0cell)
            time_calibrated[chn] = self.ch_time_bins.copy()

        return time_calibrated

    def _detector_trigger(self, time_calibrated_bins, voltage_calibrations, f=0.2):
        """cherenkov channel name"""
        det_name = 'cherenkov'
        det_time_bins = time_calibrated_bins[self.ch_names[det_name]]
        self.buffer[:] = voltage_calibrations[self.ch_names[det_name]]

        baseline = np.mean(self.buffer[(det_time_bins > 100) & (det_time_bins < 500)])
        # baseline = np.mean(self.buffer[100:200])
        bl_edge = np.argmin(det_time_bins < 100)

        self.buffer[:bl_edge] = np.min(self.buffer)
        trg_t, trg_v = linear_interpolate_trigger(det_time_bins, self.buffer, baseline, f=f)

        return trg_t, trg_v

    def _cherenkov_energy_signal(self, time_calibrated_bins, voltage_calibrations, method="both"):
        """Get cherenkov energy signal"""
        if method not in ("peak", "integral", "both"):
            ValueError("{m} method not in allowed lfs energy methods: peak, integral")
        det_name = "cherenkov"
        det_time_bins = time_calibrated_bins[self.ch_names[det_name]]
        self.buffer[:] = voltage_calibrations[self.ch_names[det_name]]
        # (d)elay (s)hift from cables. If time_calibrated bins is shifted, have to shift back for finding baseline

        pval, ival = (0, 0)

        baseline_vals = np.mean(self.buffer[(det_time_bins > 100) & (det_time_bins < 500)])
        baseline = np.mean(baseline_vals)
        self.buffer[:np.argmin(det_time_bins < 100)] = np.min(self.buffer)  # remove spikes at start

        peak_idx = np.argmax(self.buffer)
        peak_time = det_time_bins[peak_idx]

        if method != "integral":
            pval = np.max(self.buffer-baseline)
            if method == "peak":
                return pval, peak_time, baseline  # return integral/peak, argmax, baseline

        wf = self.buffer - baseline
        threshold = 3 * np.std(baseline_vals)  # positive polarity assumed
        try:
            integration_low_idx = peak_idx - np.argmin(wf[:peak_idx][::-1] >= threshold)
        except Exception as e:
            print(e)
            print("Event ID failure: ", self.event_number)

        integration_hi_idx = peak_idx + np.argmin(wf[peak_idx:] >= threshold)

        intg_time_bins = det_time_bins[integration_low_idx+1:integration_hi_idx+1] \
                             - det_time_bins[integration_low_idx:integration_hi_idx]
        intg_vals = self.buffer[integration_low_idx:integration_hi_idx]
        ival = np.sum(intg_time_bins * intg_vals)  # volts * nanoseconds
        if method == "integral":
            return ival, peak_time, baseline  # return integral/peak, argmax, baseline

        return (pval, ival), peak_time, baseline   # both

    def process_sipm_evt(self, d_f=0.2, increment_to_next_event=True):
        """Method to generate dictionary of event data. Delay NOT included. Both peak and integral values provided.
        This method is for collating SIPM/Chage Induction data sets """
        board = self.board_ids[0]
        channels = self.channels[board]
        voltage_calibrated = self.event_voltage_calibrate(board, channels)
        time_calibrated_bins = self.event_timing_calibrate(board, channels)

        det_trig, det_voltage_at_trig = self._detector_trigger(time_calibrated_bins, voltage_calibrated, f=d_f)
        (cher_en_peak, cher_en_integral), lfs_en_peak_time, lfs_en_baseline = \
            self._cherenkov_energy_signal(time_calibrated_bins, voltage_calibrated, method="both")

        ret_dict = {'drs4_evt_id': self.event.event_id,
                    "det_trig": det_trig, "det_v_at_trig": det_voltage_at_trig,
                    "det_en_peak": cher_en_peak, "det_en_integral": cher_en_integral}
        if increment_to_next_event:
            self.event = next(self.f)  # next event
            self.event_number += 1
        return ret_dict

    def test_det_points(self):
        board = self.board_ids[0]
        channels = self.channels[board]
        voltage_calibrated = self.event_voltage_calibrate(board, channels)
        time_calibrated_bins = self.event_timing_calibrate(board, channels)

        det_trig, det_voltage_at_trig = self._detector_trigger(time_calibrated_bins, voltage_calibrated)
        lfs_en_peak, lfs_en_peak_time, lfs_en_baseline = \
            self._cherenkov_energy_signal(time_calibrated_bins, voltage_calibrated, method="peak")

        fig, ax = plt.subplots(1, 1)
        labels = ["RF", "LFS fast", "Cherenkov", "T0 (inv.)"]

        for chn in channels:  # voltage and plotting
            polarity = 1
            ax.plot(time_calibrated_bins[chn], voltage_calibrated[chn] * polarity,
                    label=labels[chn-1])
        print("det_trig: ", det_trig)
        # E = Only enable for single cherenkov plot for IEEE oral
        ax.plot(det_trig, det_voltage_at_trig, "8")
        ax.plot(lfs_en_peak_time, lfs_en_peak + lfs_en_baseline, "x")
        ax.set_xlabel('time (ns)',  fontsize=18)  # E
        ax.set_ylabel('amplitude (V)',  fontsize=18)  # E
        ax.tick_params(axis='both', labelsize=16)  # E
        # ax.set_xlim((0, 200))
        ax.legend(loc='best')
        plt.show()


def test_triggers(fname):  # no det field, only LFS files here
    tst = SiPM1(fname)
    print(tst.f.board_ids)

    skip = 298
    for _ in np.arange(skip):
        tst.event = next(tst.f)
        # print(tst.event.timestamp)

    n_test = 5

    for _ in np.arange(n_test):
        tst.test_det_points()
        tst.event = next(tst.f)


def main():
    import os
    from pathlib import Path

    # single processing
    # remove below, these are all 4 channel
    # data_file_name = "20221017_Crocker_31.6V_cherenkov_500pa_DualDataset_nim_amp_p0_v9.dat"  # p0 Cher
    # data_file_name = "20221017_Crocker_31.6V_cherenkov_500pa_DualDataset_nim_amp_p2_v10.dat"  # p2 cherenkov
    # data_file_name = "20221017_Crocker_31.6V_cherenkov_500pa_DualDataset_nim_amp_p4_v11.dat"  # p4 Cher
    # data_file_name = "20221017_Crocker_31.6V_cherenkov_500pa_DualDataset_nim_amp_p5_v14.dat"  # p5 Cher
    # data_file_name = "20221017_Crocker_31.6V_cherenkov_500pa_DualDataset_nim_amp_p6_v12.dat"  # p6 Cher
    # data_file_name = "20221017_Crocker_31.6V_cherenkov_500pa_DualDataset_nim_amp_p8_v13.dat"  # p8 Cher
    data_file_name = "20221031_Davis_40_5V_Thr20mV_IEEE_Na22_DualDataset.dat"  # Na-22 Data Set 1, det 2, SiPM
    det = "cherenkov"

    fname = os.path.join(str(Path(os.getcwd()).parents[1]), "sample_data", "drs4", data_file_name)
    # p0 cherenkov event 13 (skip 12, n=1) used for plot in presentation

    test_triggers(fname)
    # energy_spectrum(fname)


if __name__ == "__main__":
    main()

