from binio import DRS4BinaryFile
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt


class CrockerSignals(object):

    def __init__(self, filename):
        self.f = DRS4BinaryFile(filename)
        # aliases
        self.n_boards = self.f.num_boards
        self.board_ids = self.f.board_ids
        self.time_widths = self.f.time_widths
        self.channels = self.f.channels
        self.n_channels = [len(self.channels[b]) for b in self.board_ids]
        self.event = next(self.f)  # first event
        self.ch_time_bins = np.zeros(1024)  # temporary working memory for time calibration
        self.t0_channel_copy = np.zeros(1024)  # temporary t0 channel copy
        # self.ch_names = ["rf", "lfs", "cherenkov", "t0"]
        self.ch_names = {"rf": 1, "lfs": 2, "cherenkov": 3, "t0": 4}

    def event_voltage_calibrate(self, board, chns):
        voltage_calibrated = {}
        for chn in chns:
            voltage_calibrated[chn] = (self.event.adc_data[board][chn] / 65536) + \
                                      (self.event.range_center / 1000) - 0.5
        # return (self.event.adc_data[board][chns] / 65536) + (self.event.range_center / 1000) - 0.5
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

    def _rf_ref_points(self, time_calibrated_bins, voltage_calibrations):
        """Find zero crossings of RF signal. Returns linear interpolated time points.
        Needs time and voltage calibrated RF data points. Also returns sign of slope from interpolation"""
        rf_waveform = voltage_calibrations[self.ch_names["rf"]]
        zero_crossings = np.where(np.diff(np.sign(rf_waveform)))[0]
        # these are indices before zero
        time_left_of_zeros = time_calibrated_bins[self.ch_names["rf"]][zero_crossings]
        time_right_of_zeros = time_calibrated_bins[self.ch_names["rf"]][zero_crossings + 1]

        v_signals_left_of_zeros = rf_waveform[zero_crossings]
        v_signals_right_of_zeros = rf_waveform[zero_crossings + 1]
        crossing_slopes = (v_signals_right_of_zeros - v_signals_left_of_zeros) / \
                          (time_right_of_zeros - time_left_of_zeros)
        # + slope -> negative to positive, - slope -> positive to negative
        # (v(0) - v(left))/(t(0)-t(left)) =m
        # -> v(0) - v(left) = m (t(0) - t(left)) -> t(0) = -v(left)/m + t(left)
        return time_left_of_zeros - (v_signals_left_of_zeros/crossing_slopes), np.sign(crossing_slopes)

    def _t0_ref_points(self, time_calibrated_bins, voltage_calibrations, f=0.2, thr=0.01):
        """Finds trigger time of each pulse for t0. Defined as a fraction f of the maximum pulse height."""
        # t0_waveform = -voltage_calibrations[self.ch_names["t0"]]  # flip to find peaks not troughs
        # 30 samples before max, 120 is after max to "baseline", 0.01 V (10 mV) threshold?
        # mask out 100 samples before and after
        t0_time_bins = time_calibrated_bins[self.ch_names["t0"]]
        self.t0_channel_copy[:] = -voltage_calibrations[self.ch_names["t0"]]

        t0_ref_time = np.ones(5) * -1
        t0_ref_voltage = np.ones(5) * -1
        # below_threshold = 0

        n_max = 5  # 200 ns range, 44.4 ns period
        n_pulse = 1

        find_pulses = True

        while (n_pulse <= n_max) & find_pulses:
            next_trigger_idx = np.argmax(self.t0_channel_copy)  # possible
            next_max_value = self.t0_channel_copy[next_trigger_idx]  # possible

            mask_left_idx = next_trigger_idx - 100
            mask_right_idx = next_trigger_idx + 100

            if mask_left_idx < 0:  # pulse near left edge
                if next_trigger_idx - 40 > 0:  # not too close for baseline subtraction
                    baseline = np.mean(self.t0_channel_copy[(next_trigger_idx-40):(next_trigger_idx-30)])
                    mask_left_idx = 0
                else:  # too close to edge for baseline subtraction and thus timing
                    self.t0_channel_copy[:next_trigger_idx + 100] = np.min(self.t0_channel_copy)
                    continue
            else:
                baseline = np.mean(self.t0_channel_copy[(next_trigger_idx - 100):(next_trigger_idx - 30)])

            if mask_right_idx > (self.t0_channel_copy.size-1):  # pulse near right edge
                mask_right_idx = self.t0_channel_copy.size  # this is so confusing

            if (next_max_value - baseline) < thr:  # no more peaks above threshold, time to stop
                find_pulses = False
                continue

            window_signal_voltage = self.t0_channel_copy[mask_left_idx:mask_right_idx] - baseline
            window_signal_tbins = t0_time_bins[mask_left_idx:mask_right_idx]

            max_sig_fv = f * (next_max_value - baseline)  # fraction of max value
            after_trig_idx = np.argmax(np.sign(window_signal_voltage - max_sig_fv))

            after_trig_v = window_signal_voltage[after_trig_idx]
            after_trig_t = window_signal_tbins[after_trig_idx]

            before_trig_v = window_signal_voltage[after_trig_idx-1]
            before_trig_t = window_signal_tbins[after_trig_idx-1]

            m = (after_trig_v - before_trig_v) / (after_trig_t - before_trig_t)  # slope
            # -> v(f) - v(l) = m (t(f) - t(l)) -> t(f) = (v(f) - v(l))/m + t(l)
            trig_t = ((max_sig_fv - before_trig_v)/m) + before_trig_t

            t0_ref_time[n_pulse - 1] = trig_t  # python index by 0...
            t0_ref_voltage[n_pulse - 1] = max_sig_fv

            self.t0_channel_copy[mask_left_idx:mask_right_idx] = np.min(self.t0_channel_copy)
            print("n_pulse: ", n_pulse)
            n_pulse += 1

        return t0_ref_time[t0_ref_time > 0], t0_ref_voltage[t0_ref_voltage > 0] + baseline

    def _detector_trigger(self, time_calibrated_bins, voltage_calibrations, f=0.2):
        """cherenkov or lfs_en channel name"""
        det_name = [key for key in self.ch_names.keys() if key not in ('rf', 'lfs', 't0')][0]
        det_time_bins = time_calibrated_bins[self.ch_names[det_name]]
        det_waveform = voltage_calibrations[self.ch_names[det_name]].copy()
        print("det name: ", det_name)
        baseline = 0
        if det_name == "cherenkov":
            baseline = np.mean(det_waveform[100:200])
            bl_edge = 100
        elif det_name == "lfs_en":
            baseline = np.mean(det_waveform[100:200])
            bl_edge = 100

        det_waveform -= baseline
        det_waveform[:bl_edge] = np.min(det_waveform)

        max_sig_fv = f * np.max(det_waveform)  # fraction of max value
        print("max_sig_fv: ", max_sig_fv)

        at_idx = np.argmax(np.sign(det_waveform - max_sig_fv))  # (a)fter (t)rigger index
        print("at_idx: ", at_idx)
        print("wavweforms 5 before and after trig: ", det_waveform[at_idx-5:at_idx+5])
        m = (det_waveform[at_idx] - det_waveform[at_idx-1])/(det_time_bins[at_idx]-det_time_bins[at_idx-1])
        trig_t = ((max_sig_fv - det_waveform[at_idx-1]) / m) + det_time_bins[at_idx-1]
        print("trig_t: ", trig_t)
        return trig_t, max_sig_fv + baseline

    def test_rf_t0_points(self):
        board = self.board_ids[0]
        channels = self.channels[board]
        voltage_calibrated = self.event_voltage_calibrate(board, channels)
        time_calibrated_bins = self.event_timing_calibrate(board, channels)

        crossings, slope_sign = self._rf_ref_points(time_calibrated_bins, voltage_calibrated)
        t0_trigs, t0_voltage_at_trig = self._t0_ref_points(time_calibrated_bins, voltage_calibrated)
        det_trig, det_voltage_at_trig = self._detector_trigger(time_calibrated_bins, voltage_calibrated)

        fig, ax = plt.subplots(1, 1)
        for chn in channels:  # voltage and plotting
            polarity = 1
            if chn == self.ch_names["t0"]:
                polarity = -1
            ax.plot(time_calibrated_bins[chn], voltage_calibrated[chn] * polarity)
            # ax.plot(voltage_calibrated[chn] * polarity)
        ax.plot(crossings, np.zeros(crossings.size), "kX")
        ax.plot(t0_trigs, t0_voltage_at_trig, "o")
        ax.plot(det_trig, det_voltage_at_trig, "8")
        ax.set_xlabel('time (ns)')
        ax.set_ylabel('amplitude (V)')
        plt.show()


def main():
    import os
    from pathlib import Path

    data_file_name = "20221017_Crocker_31.6V_cherenkov_500pa_DualDataset_nim_amp_p2_v10.dat"
    fname = os.path.join(str(Path(os.getcwd()).parents[1]), "sample_data", "drs4", data_file_name)
    tst = CrockerSignals(fname)
    print(tst.f.board_ids)

    skip = 208
    for _ in np.arange(skip):
        event = next(tst.f)
        # print(event.timestamp)

    event = next(tst.f)
    tst.test_rf_t0_points()


if __name__ == "__main__":
    main()
