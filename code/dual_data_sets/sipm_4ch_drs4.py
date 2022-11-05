from binio import DRS4BinaryFile
import numpy as np
import matplotlib.pyplot as plt
from dds_utils import *
# copy-pasted crocker_energy_timing_lfs.py
# working on: 10/31


class SiPM4(object):

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
        self.ch_time_bins = np.zeros(1024)  # temporary working memory for time calibration data
        self.buffer = np.zeros(1024)  # temporary working memory for voltage calibration data
        # self.ch_names = ["rf", "lfs", "cherenkov", "t0"]
        # 12.81
        self.cable_delays = {1: 0, 2: 6.58, 3: 0.7, 4: 12.4}  # 25.1 used  for everything but plotting. 12.81 otherwise
        self.det_type = "cherenkov"  # really trigger channel
        self.ch_names = {"rf": 1, "lfs": 2, "cherenkov": 3, "t0": 4}
        self.ch_numbers = {1: "rf", 2: "lfs", 3: "cherenkov", 4: "t0"}

    def event_voltage_calibrate(self, board, chns, verbose=False):
        voltage_calibrated = {}
        for chn in chns:
            voltage_calibrated[chn] = (self.event.adc_data[board][chn] / 65536) + \
                                      (self.event.range_center / 1000) - 0.5
        return voltage_calibrated

    def event_timing_calibrate(self, board, chns, ref=1, delay_correct=False):  # check ref is in chns before this gets called
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
            if delay_correct:
                time_calibrated[chn] = self.ch_time_bins.copy() - self.cable_delays[chn]
            else:
                time_calibrated[chn] = self.ch_time_bins.copy()

        return time_calibrated

    def _rf_ref_points(self, time_calibrated_bins, voltage_calibrations):
        """Find zero crossings of RF signal. Returns linear interpolated time points.
        Needs time and voltage calibrated RF data points. Also returns sign of slope from interpolation"""
        rf_waveform = voltage_calibrations[self.ch_names["rf"]]
        zero_crossings = np.where(np.diff(np.sign(rf_waveform)))[0] # index is BEFORE zero crossing
        time_left_of_zeros = time_calibrated_bins[self.ch_names["rf"]][zero_crossings]
        time_right_of_zeros = time_calibrated_bins[self.ch_names["rf"]][zero_crossings + 1]

        v_signals_left_of_zeros = rf_waveform[zero_crossings]
        v_signals_right_of_zeros = rf_waveform[zero_crossings + 1]
        crossing_slopes = (v_signals_right_of_zeros - v_signals_left_of_zeros) / \
                          (time_right_of_zeros - time_left_of_zeros)
        return time_left_of_zeros - (v_signals_left_of_zeros/crossing_slopes), np.sign(crossing_slopes)

    def _t0_ref_points(self, time_calibrated_bins, voltage_calibrations, f=np.array([0.2]), thr=0.01,
                       ret_max_instead=False):
        """Finds trigger time of each pulse for t0. Defined as a fraction f of the maximum pulse height."""
        t0_time_bins = time_calibrated_bins[self.ch_names["t0"]]
        # god rid of some checks here
        self.buffer[:] = -1 * voltage_calibrations[self.ch_names["t0"]]  # inverted to look for rise (and not fall) times

        t0_ref_time = np.ones(5) * -10
        t0_ref_voltage = np.ones(5) * -10

        n_max = 5  # 200 ns range, 44.4 ns period
        n_pulse = 1

        find_pulses = True

        while (n_pulse <= n_max) & find_pulses:
            next_trigger_idx = np.argmax(self.buffer)  # possible
            next_max_value = self.buffer[next_trigger_idx]  # possible

            mask_left_idx = next_trigger_idx - 100
            mask_right_idx = next_trigger_idx + 100

            if mask_left_idx < 0:  # pulse near left edge
                if next_trigger_idx - 40 > 0:  # not too close for baseline subtraction
                    baseline = np.mean(self.buffer[(next_trigger_idx-40):(next_trigger_idx-30)])
                    mask_left_idx = 0
                else:  # too close to edge for baseline subtraction and thus timing
                    self.buffer[:next_trigger_idx + 100] = np.min(self.buffer)
                    continue
            else:
                baseline = np.mean(self.buffer[(next_trigger_idx - 100):(next_trigger_idx - 30)])

            if mask_right_idx > (self.buffer.size-1):  # pulse near right edge
                mask_right_idx = self.buffer.size  # this is so confusing

            # above already checks for left or right edge

            if (n_pulse > 1) & (np.sum(np.abs(t0_time_bins[next_trigger_idx] - t0_ref_time[:n_pulse - 1]) < 35) > 0):
                # print("Saw this!")
                find_pulses = False
                continue  # This guesses that triggering on noise between pulses, so stop

            if (next_max_value - baseline) < thr:  # no more peaks above threshold, time to stop
                find_pulses = False
                continue  # effectively a break

            window_signal_voltage = self.buffer[mask_left_idx:mask_right_idx]
            window_signal_tbins = t0_time_bins[mask_left_idx:mask_right_idx]

            trg_t, trg_v = linear_interpolate_trigger(window_signal_tbins, window_signal_voltage, baseline, f=f,
                                                      ret_max_instead=ret_max_instead)

            t0_ref_time[n_pulse - 1] = trg_t  # python index by 0...
            t0_ref_voltage[n_pulse - 1] = trg_v

            self.buffer[mask_left_idx:mask_right_idx] = np.min(self.buffer)
            n_pulse += 1

        # return t0_ref_time[t0_ref_time > -10], t0_ref_voltage[t0_ref_voltage > -10]
        return t0_ref_time, t0_ref_voltage

    def _detector_trigger(self, time_calibrated_bins, voltage_calibrations, f=0.2):
        """cherenkov channel name"""
        det_name = 'cherenkov'
        det_time_bins = time_calibrated_bins[self.ch_names[det_name]]
        self.buffer[:] = voltage_calibrations[self.ch_names[det_name]]

        baseline = np.mean(self.buffer[100:200])
        bl_edge = 100

        self.buffer[:bl_edge] = np.min(self.buffer)
        trg_t, trg_v = linear_interpolate_trigger(det_time_bins, self.buffer, baseline, f=f)

        return trg_t, trg_v

    def _cherenkov_energy_signal(self, time_calibrated_bins, voltage_calibrations, method="both", delay_corrected=False):
        """Get cherenkov energy signal"""
        if method not in ("peak", "integral", "both"):
            ValueError("{m} method not in allowed lfs energy methods: peak, integral")
        det_name = "cherenkov"
        det_time_bins = time_calibrated_bins[self.ch_names[det_name]]
        self.buffer[:] = voltage_calibrations[self.ch_names[det_name]]
        ds = delay_corrected * self.cable_delays[self.ch_names[det_name]]
        # (d)elay (s)hift from cables. If time_calibrated bins is shifted, have to shift back for finding baseline

        peak_idx = np.argmax(self.buffer)
        peak_time = det_time_bins[peak_idx]

        pval, ival = (0, 0)
        if method != "integral":
            baseline = np.mean(self.buffer[((det_time_bins + ds) > 1) & ((det_time_bins + ds) < 8)])  # 1-8 ns used for baseline
            pval = np.max(self.buffer-baseline)
            if method == "peak":
                return pval, peak_time, baseline  # return integral/peak, argmax, baseline

        baseline_vals = self.buffer[((det_time_bins + ds) > 1) & ((det_time_bins + ds) < 8)]
        baseline = np.mean(baseline_vals)
        wf = self.buffer - baseline
        threshold = 3 * np.std(baseline_vals) # positive polarity assumed
        integration_low_idx = peak_idx - np.argmin(wf[:peak_idx][::-1] >= threshold)
        integration_hi_idx = peak_idx + np.argmin(wf[peak_idx:] >= threshold)

        intg_time_bins = det_time_bins[integration_low_idx+1:integration_hi_idx+1] \
                             - det_time_bins[integration_low_idx:integration_hi_idx]
        intg_vals = self.buffer[integration_low_idx:integration_hi_idx]
        ival = np.sum(intg_time_bins * intg_vals)  # volts * nanoseconds
        if method == "integral":
            return ival, peak_time, baseline  # return integral/peak, argmax, baseline

        return (pval, ival), peak_time, baseline   # both

    def process_sipm_evt(self, t0_f=np.array([0.2]), d_f=0.2):
        """Method to generate dictionary of event data. Delay NOT included. Both peak and integral values provided.
        This method is for collating SIPM/Chage Induction data sets """
        board = self.board_ids[0]
        channels = self.channels[board]
        voltage_calibrated = self.event_voltage_calibrate(board, channels)
        time_calibrated_bins = self.event_timing_calibrate(board, channels)

        crossings, slope_sign = self._rf_ref_points(time_calibrated_bins, voltage_calibrated)
        t0_trigs, t0_voltage_at_trig = self._t0_ref_points(time_calibrated_bins, voltage_calibrated, f=t0_f)
        det_trig, det_voltage_at_trig = self._detector_trigger(time_calibrated_bins, voltage_calibrated, f=d_f)
        (cher_en_peak, cher_en_integral), lfs_en_peak_time, lfs_en_baseline = \
            self._cherenkov_energy_signal(time_calibrated_bins, voltage_calibrated, method="both", delay_corrected=False)

        ret_dict = {"rf_zero_crossings": crossings, "rf_zero_crossing_signs": slope_sign,
                    "t0_trigs": t0_trigs, "t0_voltage_at_trig": t0_voltage_at_trig,
                    "det_trig": det_trig, "det_voltage_at_trig": det_voltage_at_trig,
                    "det_en_peak": cher_en_peak, "det_en_integral": cher_en_integral}

        return ret_dict

    def test_rf_t0_points(self, delay=False):
        board = self.board_ids[0]
        channels = self.channels[board]
        voltage_calibrated = self.event_voltage_calibrate(board, channels)
        time_calibrated_bins = self.event_timing_calibrate(board, channels)

        crossings, slope_sign = self._rf_ref_points(time_calibrated_bins, voltage_calibrated)
        t0_trigs, t0_voltage_at_trig = self._t0_ref_points(time_calibrated_bins, voltage_calibrated)
        det_trig, det_voltage_at_trig = self._detector_trigger(time_calibrated_bins, voltage_calibrated)
        lfs_en_peak, lfs_en_peak_time, lfs_en_baseline = \
            self._cherenkov_energy_signal(time_calibrated_bins, voltage_calibrated, method="peak")

        print("t0_trigs: ", t0_trigs)

        fig, ax = plt.subplots(1, 1)
        labels = ["RF", "LFS fast", "Cherenkov", "T0 (inv.)"]

        for chn in channels:  # voltage and plotting
            # if self.ch_numbers[chn] != "cherenkov":  # To only plot cherenkov
            #     continue
            polarity = 1
            if chn == self.ch_names["t0"]:
                polarity = -1  # needs to be flipped for cherenkov, not flipped for lfs trigger
            ax.plot(time_calibrated_bins[chn] - (delay * self.cable_delays[chn]), voltage_calibrated[chn] * polarity,
                    label=labels[chn-1])
        print("det_trig: ", det_trig)
        # E = Only enable for single cherenkov plot for IEEE oral
        ax.plot(crossings, np.zeros(crossings.size), "kX")
        ax.plot(t0_trigs[t0_trigs > -10], t0_voltage_at_trig[t0_voltage_at_trig > -10], "o")
        ax.plot(det_trig, det_voltage_at_trig, "8")
        ax.plot(lfs_en_peak_time, lfs_en_peak + lfs_en_baseline, "x")
        ax.set_xlabel('time (ns)',  fontsize=18)  # E
        ax.set_ylabel('amplitude (V)',  fontsize=18)  # E
        ax.tick_params(axis='both', labelsize=16)  # E
        # ax.set_xlim((0, 200))
        ax.legend(loc='best')
        plt.show()


def test_triggers(fname):  # no det field, only LFS files here
    tst = SiPM4(fname)
    print(tst.f.board_ids)

    skip = 22
    for _ in np.arange(skip):
        tst.event = next(tst.f)
        # print(tst.event.timestamp)

    n_test = 1

    delay = False
    for _ in np.arange(n_test):
        tst.test_rf_t0_points(delay=delay)
        tst.event = next(tst.f)


def main():
    import os
    from pathlib import Path

    # single processing
    data_file_name = "20221017_Crocker_31.6V_cherenkov_500pa_DualDataset_nim_amp_p0_v9.dat"  # p0 Cher
    # data_file_name = "20221017_Crocker_31.6V_cherenkov_500pa_DualDataset_nim_amp_p2_v10.dat"  # p2 cherenkov
    # data_file_name = "20221017_Crocker_31.6V_cherenkov_500pa_DualDataset_nim_amp_p4_v11.dat"  # p4 Cher
    # data_file_name = "20221017_Crocker_31.6V_cherenkov_500pa_DualDataset_nim_amp_p5_v14.dat"  # p5 Cher
    # data_file_name = "20221017_Crocker_31.6V_cherenkov_500pa_DualDataset_nim_amp_p6_v12.dat"  # p6 Cher
    # data_file_name = "20221017_Crocker_31.6V_cherenkov_500pa_DualDataset_nim_amp_p8_v13.dat"  # p8 Cher
    det = "cherenkov"

    fname = os.path.join(str(Path(os.getcwd()).parents[1]), "sample_data", "drs4", data_file_name)
    # p0 cherenkov event 13 (skip 12, n=1) used for plot in presentation

    test_triggers(fname)
    # energy_spectrum(fname)


if __name__ == "__main__":
    main()

