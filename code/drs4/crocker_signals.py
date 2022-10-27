from binio import DRS4BinaryFile
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
# This file was to test trigger method implementation. Analysis done in crocker_signals_timing

def linear_interpolate_trigger(time_bins, waveform, baseline, f=0.2):
    """Assumes positive polarity signals"""
    wf = waveform - baseline
    t = time_bins
    max_sig_fv = f * np.max(wf)

    ati = np.argmax(np.sign(wf - max_sig_fv))  # (a)fter (t)rigger (i)ndex
    m = (wf[ati] - wf[ati-1]) / (t[ati]-t[ati-1])  # slope
    interp_t = t[ati-1] + ((max_sig_fv - wf[ati-1])/m)

    return interp_t, max_sig_fv + baseline  # add back baseline for plotting


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


class CrockerSignalsCherenkov(object):

    def __init__(self, filename):
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
        self.ch_names = {"rf": 1, "lfs": 2, "cherenkov": 3, "t0": 4}

    def event_voltage_calibrate(self, board, chns):
        voltage_calibrated = {}
        for chn in chns:
            # voltage_calibrated[chn] = (self.event.adc_data[board][chn] / 65536) + \
            #                           (self.event.range_center / 1000) - 0.5
            wf_adc_to_v = (self.event.adc_data[board][chn] / 65536) + \
                                      (self.event.range_center / 1000) - 0.5

            if "lfs_en" in self.ch_names.keys():  # remove spikes
                if (np.argmax(voltage_calibrated[chn]) < 10) & (np.max(voltage_calibrated[chn]) > 0.5):
                    # spike near beginning of trace, exceeds RF amplitude limits
                    first_10 = wf_adc_to_v[:10]
                    try:
                        wf_adc_to_v[:10] = np.mean(first_10 < 0.5)
                    except:
                        print("Wide spike found at beginning of a {c} trace!".format(c=self.ch_names[chn]))
                        wf_adc_to_v[:10] = np.mean(wf_adc_to_v[10:20] < 0.5)

                if (np.argmax(voltage_calibrated[chn]) >= (wf_adc_to_v.size - 10)) & (np.max(voltage_calibrated[chn]) > 0.5):
                    # spike near end of trace, exceeds RF amplitude limits
                    last_10 = wf_adc_to_v[-10:]
                    try:
                        wf_adc_to_v[-10:] = np.mean(last_10 < 0.5)
                    except:
                        print("Wide spike found at end of a {c} trace!".format(c=self.ch_names[chn]))
                        wf_adc_to_v[-10:] = np.mean(wf_adc_to_v[-20:-10] < 0.5)
            voltage_calibrated[chn] = wf_adc_to_v
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
        self.buffer[:] = -voltage_calibrations[self.ch_names["t0"]]

        t0_ref_time = np.ones(5) * -10
        t0_ref_voltage = np.ones(5) * -10
        # below_threshold = 0

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

            if (next_max_value - baseline) < thr:  # no more peaks above threshold, time to stop
                find_pulses = False
                continue

            window_signal_voltage = self.buffer[mask_left_idx:mask_right_idx]
            window_signal_tbins = t0_time_bins[mask_left_idx:mask_right_idx]

            trg_t, trg_v = linear_interpolate_trigger(window_signal_tbins, window_signal_voltage, baseline, f=f)

            t0_ref_time[n_pulse - 1] = trg_t  # python index by 0...
            t0_ref_voltage[n_pulse - 1] = trg_v

            self.buffer[mask_left_idx:mask_right_idx] = np.min(self.buffer)
            # print("n_pulse: ", n_pulse)
            n_pulse += 1

        return t0_ref_time[t0_ref_time > -10], t0_ref_voltage[t0_ref_voltage > -10]

    def _detector_trigger(self, time_calibrated_bins, voltage_calibrations, f=0.2):
        """cherenkov or lfs_en channel name"""
        det_name = [key for key in self.ch_names.keys() if key not in ('rf', 'lfs', 't0')][0]
        det_time_bins = time_calibrated_bins[self.ch_names[det_name]]
        self.buffer[:] = voltage_calibrations[self.ch_names[det_name]]
        print("det name: ", det_name)
        baseline = 0
        bl_edge = 100
        if det_name == "cherenkov":
            baseline = np.mean(self.buffer[100:200])
            bl_edge = 100
        elif det_name == "lfs_en":
            baseline = np.mean(self.buffer[100:200])
            bl_edge = 100

        self.buffer[:bl_edge] = np.min(self.buffer)
        trg_t, trg_v = linear_interpolate_trigger(det_time_bins, self.buffer, baseline, f=f)
        return trg_t, trg_v

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

    data_file_name = "20221017_Crocker_31.6V_cherenkov_500pa_DualDataset_nim_amp_p2_v10.dat"  # cherenkov
    # data_file_name = "20221017_Crocker_31.6V_LFS_500pa_SingleDataset_nim_amp_p2_v19.dat"  # LFS
    fname = os.path.join(str(Path(os.getcwd()).parents[1]), "sample_data", "drs4", data_file_name)
    tst = CrockerSignalsCherenkov(fname)
    print(tst.f.board_ids)

    skip = 308
    for _ in np.arange(skip):
        event = next(tst.f)
        # print(event.timestamp)

    event = next(tst.f)
    tst.test_rf_t0_points()


if __name__ == "__main__":
    main()
