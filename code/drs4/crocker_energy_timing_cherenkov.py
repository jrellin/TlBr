from binio import DRS4BinaryFile
import numpy as np
import matplotlib.pyplot as plt
from crocker_utils import *
# nearly identical to crocker_signals_timing_lfs.py, but 2D energy-time plots where time is just a projection
# working on: 10/30
# TODO: This has not been editted yet to work with cherenkov signals. Also 2D plotting


class CrockerSignals(object):

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
        self.cable_delays = {1: 0, 2: 6.58, 3: 0.7, 4: 25.1}  # 24.81 is money, LFS only for this file
        self.det_type = "lfs_en"  # really trigger channel
        self.ch_names = {"rf": 1, "lfs": 2, "lfs_en": 3, "t0": 4}

    def event_voltage_calibrate(self, board, chns, verbose=False):
        voltage_calibrated = {}
        for chn in chns:
            wf_adc_to_v = (self.event.adc_data[board][chn] / 65536) + \
                                      (self.event.range_center / 1000) - 0.5

            if "lfs_en" in self.ch_names.keys():  # remove spikes
                first_10 = wf_adc_to_v[:10]
                last_10 = wf_adc_to_v[-10:]
                if np.sum(first_10 > 0.5):  # if any larger than 0.5
                    if verbose:
                        print("Spike at beginning in channel {c}".format(c=chn))
                    try:
                        wf_adc_to_v[:10] = np.mean(first_10[first_10 < 0.5])
                    except:
                        wf_adc_to_v[:10] = np.mean((wf_adc_to_v[10:20])[wf_adc_to_v[10:20] < 0.5])
                if np.sum(last_10 > 0.5):  # if any larger than 0.5
                    if verbose:
                        print("Spike at end of channel {c}".format(c=chn))
                    last_10 = wf_adc_to_v[-10:]
                    try:
                        wf_adc_to_v[-10:] = np.mean(last_10[last_10 < 0.5])
                    except:
                        wf_adc_to_v[-10:] = np.mean((wf_adc_to_v[-20:-10])[wf_adc_to_v[-20:-10] < 0.5])
            voltage_calibrated[chn] = wf_adc_to_v
        return voltage_calibrated

    def event_timing_calibrate(self, board, chns, ref=1, delay_correct=False):  # check ref is in chns before this gets called
        # TODO: Add delay option
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

    def _t0_ref_points(self, time_calibrated_bins, voltage_calibrations, f=np.array([0.2]), thr=0.01):
        """Finds trigger time of each pulse for t0. Defined as a fraction f of the maximum pulse height."""
        # t0_waveform = -voltage_calibrations[self.ch_names["t0"]]  # flip to find peaks not troughs
        # 30 samples before max, 120 is after max to "baseline", 0.01 V (10 mV) threshold?
        # mask out 100 samples before and after
        t0_time_bins = time_calibrated_bins[self.ch_names["t0"]]
        polarity = -1  # cherenkov
        if self.det_type == "lfs_en":
            polarity = 1
        self.buffer[:] = polarity * voltage_calibrations[self.ch_names["t0"]]

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

            # trg_t, trg_v = linear_interpolate_trigger(window_signal_tbins, window_signal_voltage, baseline, f=f)
            trg_t, trg_v = linear_interpolate_trigger2(window_signal_tbins, window_signal_voltage, baseline, f=f)

            t0_ref_time[n_pulse - 1] = trg_t  # python index by 0...
            t0_ref_voltage[n_pulse - 1] = trg_v

            self.buffer[mask_left_idx:mask_right_idx] = np.min(self.buffer)
            n_pulse += 1

        return t0_ref_time[t0_ref_time > -10], t0_ref_voltage[t0_ref_voltage > -10]

    def _detector_trigger(self, time_calibrated_bins, voltage_calibrations, f=0.2):
        """cherenkov or lfs_en channel name"""
        if self.det_type == 'lfs_en':
            det_name = 'lfs'
        else:
            det_name = 'cherenkov'
        det_time_bins = time_calibrated_bins[self.ch_names[det_name]]
        self.buffer[:] = voltage_calibrations[self.ch_names[det_name]]
        # print("det name: ", det_name)
        if det_name == "cherenkov":
            baseline = np.mean(self.buffer[100:200])
            bl_edge = 100
        else:  # lfs
            baseline = np.mean(self.buffer[20:100])
            bl_edge = 100

        self.buffer[:bl_edge] = np.min(self.buffer)
        if det_name == "cherenkov":
            # trg_t, trg_v = linear_interpolate_trigger(det_time_bins, self.buffer, baseline, f=f)
            trg_t, trg_v = linear_interpolate_trigger2(det_time_bins, self.buffer, baseline, f=f)
        else:
            trg_t, trg_v = leading_edge_trigger(det_time_bins, self.buffer, baseline, thr=0.1)
        return trg_t, trg_v

    def _lfs_energy_signal(self, time_calibrated_bins, voltage_calibrations, method="peak", delay_corrected=False):
        """Get lfs energy signal"""
        # TODO: adjust baseline for integration method because of delays
        if method not in ("peak", "integral"):
            ValueError("{m} method not in allowed lfs energy methods: peak, integral")
        det_name = "lfs_en"
        det_time_bins = time_calibrated_bins[self.ch_names[det_name]]
        self.buffer[:] = voltage_calibrations[self.ch_names[det_name]]
        ds = delay_corrected * self.cable_delays[self.ch_names["lfs_en"]]
        # (d)elay (s)hift from cables. If time_calibrated bins is shifted, have to shift back for finding baseline

        peak_idx = np.argmax(self.buffer)
        peak_time = det_time_bins[peak_idx]

        if method == "peak":
            baseline = np.mean(self.buffer[((det_time_bins + ds) > 1) & ((det_time_bins + ds) < 8)])  # 1-8 ns used for baseline
            val = np.max(self.buffer-baseline)
        else:  # integral
            baseline_vals = self.buffer[((det_time_bins + ds) > 1) & ((det_time_bins + ds) < 8)]
            baseline = np.mean(baseline_vals)
            wf = self.buffer - baseline
            threshold = 3 * np.std(baseline_vals) # positive polarity assumed
            integration_low_idx = peak_idx - np.argmin(wf[:peak_idx][::-1] >= threshold)
            integration_hi_idx = peak_idx + np.argmin(wf[peak_idx:] >= threshold)

            intg_time_bins = det_time_bins[integration_low_idx+1:integration_hi_idx+1] \
                             - det_time_bins[integration_low_idx:integration_hi_idx]
            intg_vals = self.buffer[integration_low_idx:integration_hi_idx]
            val = np.sum(intg_time_bins * intg_vals)  # volts * nanoseconds

        return val, peak_time, baseline   # return integral/peak, argmax, baseline

    def lfs_energy_spectrum(self, method="peak", log_scale=False):
        """1D energy spectrum as a quick way to get energy. Use this method to test the amplitude/integration method"""
        board = self.board_ids[0]
        channels = self.channels[board]
        # voltage_calibrated = self.event_voltage_calibrate(board, channels)
        # time_calibrated_bins = self.event_timing_calibrate(board, channels)

        if method == "peak":
            en_counts, en_bins = np.histogram([], bins=np.linspace(0, 0.7, num=4097))
            xlabel = "Peak Voltage"
        else:  # "integral"
            en_counts, en_bins = np.histogram([], bins=np.linspace(0, 30, num=4097))
            xlabel = "Integral Value"

        evt_buffer = np.zeros(50000)
        evts = 0
        ptr = 0
        keep_reading = True
        delay_correct = True  # cable delay

        try:
            while keep_reading:
                voltage_calibrated = self.event_voltage_calibrate(board, channels)
                time_calibrated_bins = self.event_timing_calibrate(board, channels, delay_correct=True)

                val, peak_time, bl = \
                    self._lfs_energy_signal(time_calibrated_bins, voltage_calibrated,
                                            method=method, delay_corrected=delay_correct)
                # val = integral or peak depending on method, peak_time = time of voltage peak, bl = baseline
                evt_buffer[ptr] = val
                ptr += 1
                evts += 1
                if ptr >= evt_buffer.size:
                    print("Full energy buffer. Histogramming.")
                    en_counts += np.histogram(evt_buffer[:ptr], bins=en_bins)[0]
                    ptr = 0
                self.event = next(self.f)
        except StopIteration:
            print("Reached last event!")
        finally:
            print("Emptying remaining buffers.")
            print("Total Events: ", evts)
            en_counts += np.histogram(evt_buffer[:ptr], bins=en_bins)[0]

            fig, ax = plt.subplots(1, 1, figsize=(16, 12))  # ax1 -> rise_times, ax2 -> amplitudes
            fig.suptitle("LFS {m} Energy Spectrum".format(m=method), fontsize=22)
            ax.step(0.5 * (en_bins[1:] + en_bins[:-1]), en_counts, 'b-', where='mid')
            ax.set_xlabel(xlabel, fontsize=18)
            ax.set_ylabel("counts", fontsize=18)
            ax.tick_params(axis='both', labelsize=16)

            if log_scale:
                ax.set_yscale('log')

            plt.show()

    def test_rf_t0_points(self):
        board = self.board_ids[0]
        channels = self.channels[board]
        voltage_calibrated = self.event_voltage_calibrate(board, channels)
        time_calibrated_bins = self.event_timing_calibrate(board, channels)

        crossings, slope_sign = self._rf_ref_points(time_calibrated_bins, voltage_calibrated)
        t0_trigs, t0_voltage_at_trig = self._t0_ref_points(time_calibrated_bins, voltage_calibrated)
        det_trig, det_voltage_at_trig = self._detector_trigger(time_calibrated_bins, voltage_calibrated)
        lfs_en_peak, lfs_en_peak_time, lfs_en_baseline = \
            self._lfs_energy_signal(time_calibrated_bins, voltage_calibrated, method="peak")
        # _lfs_energy_signal(self, time_calibrated_bins, voltage_calibrations, method="peak")
        # return val, peak_idx, baseline
        # "|"
        print("t0_trigs: ", t0_trigs)

        fig, ax = plt.subplots(1, 1)
        for chn in channels:  # voltage and plotting
            polarity = 1
            if (chn == self.ch_names["t0"]) & (self.det_type == "cherenkov"):
                polarity = -1  # needs to be flipped for cherenkov, not flipped for lfs trigger
            ax.plot(time_calibrated_bins[chn], voltage_calibrated[chn] * polarity)
        print("det_trig: ", det_trig)
        ax.plot(crossings, np.zeros(crossings.size), "kX")
        ax.plot(t0_trigs, t0_voltage_at_trig, "o")
        ax.plot(det_trig, det_voltage_at_trig, "8")
        ax.plot(lfs_en_peak_time, lfs_en_peak + lfs_en_baseline, "x")
        ax.set_xlabel('time (ns)')
        ax.set_ylabel('amplitude (V)')
        plt.show()

    def rf_to_t0_and_detector(self, log_scale=False, suppress_plots=False, save_histograms=False, save_fname=None):
        """Generates histograms of t0 - rf, detector - rf, and detector - t0"""
        board = self.board_ids[0]
        channels = self.channels[board]
        t0_frac = np.array([0.2])  # "CFD" for t0

        # TODO: LFS only section changes
        # t0_to_rf_times, t0_to_rf_bins = np.histogram([], bins=np.linspace(-1, 5, num=601)) position 2
        t0_to_rf_times, t0_to_rf_bins = np.histogram([], bins=np.linspace(-1, 5, num=601))
        # t0_to_rf_times, t_bins = np.histogram([], bins=np.linspace(-4, 44, num=1001))  # lfs original
        det_to_rf_times, t_bins = np.histogram([], bins=np.linspace(-4, 44, num=1001))
        det_to_t0_times, _ = np.histogram([], bins=t_bins)

        t0_rf_time_buffer = np.zeros(50000)  # temp storage
        det_rf_time_buffer = np.zeros(50000)
        det_t0_time_buffer = np.zeros(50000)  # add

        ptr_g, ptr_rf = 0, 0  # ptr to current point in buffer for gamma and rf, respectively
        (tot_evts, evts_used, missed_pulses) = (0, 0, 0)  # below threshold
        # tot_evts = total triggered events saved, events_used is triggers with valid cuts,
        # missed pulses is lost t0 pulses because of record length or below threshold

        keep_reading = True

        check = 1

        try:
            while keep_reading:
                voltage_calibrated = self.event_voltage_calibrate(board, channels)
                time_calibrated_bins = self.event_timing_calibrate(board, channels, delay_correct=True)

                crossings, slope_sign = self._rf_ref_points(time_calibrated_bins, voltage_calibrated)
                t0_trigs, t0_max_voltages = self._t0_ref_points(time_calibrated_bins, voltage_calibrated, f=t0_frac)
                det_trig, det_voltage_at_trig = self._detector_trigger(time_calibrated_bins, voltage_calibrated)

                # if check < 10:
                #     print("det trig: ", det_trig)
                #     print("t0_trigs: ", t0_trigs)
                #     print("t0_trigs.shape: ", t0_trigs.shape)
                #     print("subtraction: ", det_trig - t0_trigs)
                #     check += 1

                if (np.sum((det_trig - crossings) > 0) <= 0) or (np.sum((det_trig - t0_trigs) > 0) <= 0):
                    # No sensible nearest triggers
                    self.event = next(self.f)
                    missed_pulses += 5
                    tot_evts += 1
                    continue

                rf_ref_idx = ((det_trig - crossings) > 0)
                t0_ref_idx = ((det_trig - t0_trigs) > 0)

                tgamma_to_rf = (det_trig - crossings)[rf_ref_idx].min()
                try:
                    tgamma_to_t0 = (det_trig - t0_trigs)[t0_ref_idx].min()
                except:
                    print("det_trg: ", det_trig)
                    print("t0_trigs: ", t0_trigs)
                    print("t0_ref_idx: ", t0_ref_idx)

                del_t = correlate_pulse_trains(t0_trigs, crossings[slope_sign < 0])
                # t0 relative to negative slope crossing RF
                t0refs = del_t.size  # how many t0-rf pairs exist

                det_rf_time_buffer[ptr_g:ptr_g + 1] = tgamma_to_rf
                det_t0_time_buffer[ptr_g:ptr_g + 1] = tgamma_to_t0
                t0_rf_time_buffer[ptr_rf:ptr_rf + t0refs] = del_t

                ptr_g += 1  # current index into  gamma buffer
                ptr_rf += t0refs

                evts_used += 1
                tot_evts += 1
                missed_pulses += (5 - t0refs)  # ideally 5 because of crocker RF period (44.4 ns)

                if (ptr_rf + t0refs) > (t0_rf_time_buffer.size - 20):  # Next set of events close to end of buffer
                    print("Appending to rf-t0 histograms. Full rf-t0 buffers.")
                    t0_to_rf_times += np.histogram(t0_rf_time_buffer[:ptr_rf], bins=t0_to_rf_bins)[0]
                    ptr_rf = 0  # back to beginning of buffer

                if (ptr_g + 1) > (det_rf_time_buffer.size - 20):  # Next set of events close to end of buffer
                    print("Appending to gamma histograms. Full gamma buffers.")
                    det_to_rf_times += np.histogram(det_rf_time_buffer[:ptr_g], bins=t_bins)[0]
                    det_to_t0_times += np.histogram(det_t0_time_buffer[:ptr_g], bins=t_bins)[0]
                    ptr_g = 0  # back to beginning of buffer

                self.event = next(self.f)  # move to next event, stop iteration otherwise

        except StopIteration:
            print("Reached last event!")
            keep_reading = False
        finally:
            print("Emptying remaining buffers.")
            t0_to_rf_times += np.histogram(t0_rf_time_buffer[:ptr_rf], bins=t0_to_rf_bins)[0]
            det_to_rf_times += np.histogram(det_rf_time_buffer[:ptr_g], bins=t_bins)[0]
            det_to_t0_times += np.histogram(det_t0_time_buffer[:ptr_g], bins=t_bins)[0]

            print("Total (trigger) events: ", tot_evts)
            print("Total triggers used: ", evts_used)
            print("Missed pulses: ", missed_pulses)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 12))  # ax1 -> rise_times, ax2 -> amplitudes
            if "cherenkov" in self.ch_names.keys():
                det = "Cherenkov"
            else:  # LFS
                det = "LFS"
            fig.suptitle(det + " Detector, RF, and T0 $\Delta$T", fontsize=22)

            for ax, bin_edges, values, \
                xlbl, title, plot_label in zip((ax1, ax2), (t0_to_rf_bins, t_bins),
                                               (t0_to_rf_times, det_to_rf_times), ("$\Delta$T (ns)", "$\Delta$T (ns)"),
                                               ("$T_{t0}-T_{rf}$", "$\Delta$T with RF or T0"),
                                               ("t0 to RF", "$T_{\gamma}-T_{rf}$")):
                bins = (bin_edges[1:] + bin_edges[:-1]) / 2
                ax.step(bins, values, 'b-', where='mid', label=plot_label)
                ax.set_xlabel(xlbl, fontsize=18)
                ax.set_ylabel("counts", fontsize=18)
                ax.set_title(title, fontsize=18)
                ax.tick_params(axis='both', labelsize=16)

                if log_scale:
                    ax.set_yscale('log')

            t_centers = 0.5 * (t_bins[1:] + t_bins[:-1])
            # if self.det_type == "cherenkov":
            #     t_centers += 0

            ax2.step(t_centers, det_to_t0_times, 'g-', where='mid', label="$T_{\gamma}-T_{t0}$")
            ax2.legend(loc='best')
            if not suppress_plots:
                plt.show()

            if save_histograms:
                if save_fname is None:
                    save_fname = "histograms"
                np.savez(save_fname, filename=self.filename,
                         t0_to_rf_bins=t0_to_rf_bins, t0_to_rf_counts=t0_to_rf_times,
                         det_to_ref_time_bins=t_bins,
                         det_to_rf_counts=det_to_rf_times, det_to_t0_counts=det_to_t0_times)


def test_triggers(fname):  # no det field, only LFS files here
    tst = CrockerSignals(fname)
    print(tst.f.board_ids)

    skip = 0 # 1200, 3210 with p2 is interesting spike
    # 3206 with p2 illustrates possible edge case for recovering t0 pulses near edge
    for _ in np.arange(skip):
        tst.event = next(tst.f)
        # print(tst.event.timestamp)

    n_test = 1

    for _ in np.arange(n_test):
        tst.test_rf_t0_points()
        tst.event = next(tst.f)

    # tst.event = next(tst.f)
    # tst.test_rf_t0_points()


def t0_rf_det_delta_t(fname): # no det field, only LFS files here
    import os
    base_fname = os.path.splitext(fname)[0]
    print(base_fname)

    save_histograms = False
    t0data = CrockerSignals(fname)
    print(t0data.f.board_ids)
    t0data.rf_to_t0_and_detector(save_histograms=save_histograms, save_fname=base_fname)


def energy_spectrum(fname):
    tst = CrockerSignals(fname)
    print(tst.f.board_ids)

    # method = "peak"
    method = "integral"
    tst.lfs_energy_spectrum(method=method, log_scale=False)
    # lfs_energy_spectrum(self, method="peak", log_scale=False)


def main():
    import os
    from pathlib import Path

    # data_file_name = "20221017_Crocker_31.6V_cherenkov_500pa_DualDataset_nim_amp_p2_v10.dat"  # cherenkov
    # det = "cherenkov"
    data_file_name = "20221017_Crocker_31.6V_LFS_500pa_SingleDataset_nim_amp_p2_v19.dat"  # p2 LFS
    # data_file_name = "20221017_Crocker_31.6V_LFS_500pa_DualDataset_nim_amp_p8_v15.dat"  # p8 LFS
    # det = "lfs_en"  # no det field for this file

    # cherenkov triggering channels: [rf, lfs_timing, cherenkov, t0]
    # LFS triggering channels: [rf, lfs_timing, lfs_energy, t0]
    fname = os.path.join(str(Path(os.getcwd()).parents[1]), "sample_data", "drs4", data_file_name)

    test_triggers(fname)
    # t0_rf_det_delta_t(fname)
    # energy_spectrum(fname)


if __name__ == "__main__":
    main()
