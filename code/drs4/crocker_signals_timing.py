from binio import DRS4BinaryFile
import numpy as np
import matplotlib.pyplot as plt
# stopped working on 10/28


def linear_interpolate_trigger(time_bins, waveform, baseline, f=0.2):
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


def rise_time_points(time_bins, waveform, baseline, f=np.array([0.1, 0.2, 0.9])):
    """Same as linear interpolate trigger but returns multiple thresholds and the maximum relative amplitude.
     Used for t0 rise time analysis and triggers."""
    wf = waveform.copy()
    wf -= baseline
    t = time_bins

    interp_trgs = np.zeros_like(f)

    max_sig = np.max(wf)
    for i, fract in enumerate(f):
        max_sig_fv = fract * max_sig
        ati = np.argmax(np.sign(wf - max_sig_fv))  # (a)fter (t)rigger (i)ndex
        m = (wf[ati] - wf[ati - 1]) / (t[ati] - t[ati - 1])  # slope
        if m != 0:
            interp_trgs[i] = t[ati - 1] + ((max_sig_fv - wf[ati - 1]) / m)
        else:  # no need to interp
            interp_trgs[i] = t[ati - 1]

    return interp_trgs, max_sig  # relative max above baseline


class CrockerSignals(object):

    def __init__(self, filename, det="cherenkov"):
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
        # self.ch_names = {"rf": 1, "lfs": 2, "cherenkov": 3, "t0": 4}
        if det not in ("cherenkov", "lfs_en"):
            ValueError("det {d} not in required types: cherenkov, lfs_en".format(d=det))
        if det == "cherenkov":
            self.cable_delays = {1: 0, 2: 0, 3: 0, 4: 0}
        else:  # must be lfs
            # cable length delay measurement (difference was t0 in FIFO for LFS measurements)
            # Channels:  {1: 'rf', 2: 'cherenkov', 3: 'lfs', 4: 't0'}  # 2 and 3 were swapped
            # Rise Time Maximums:  {1: 86.725, 2: 87.42500000000001, 3: 93.305, 4: 99.535}
            self.cable_delays = {1: 0, 2: 6.58, 3: 0.7, 4: 12.81}
        self.det_type = det
        self.ch_names = {"rf": 1, "lfs": 2, det: 3, "t0": 4}

    def event_voltage_calibrate(self, board, chns):
        voltage_calibrated = {}
        for chn in chns:
            # voltage_calibrated[chn] = (self.event.adc_data[board][chn] / 65536) + \
            #                           (self.event.range_center / 1000) - 0.5
            wf_adc_to_v = (self.event.adc_data[board][chn] / 65536) + \
                                      (self.event.range_center / 1000) - 0.5

            if "lfs_en" in self.ch_names.keys():  # remove spikes
                first_10 = wf_adc_to_v[:10]
                last_10 = wf_adc_to_v[-10:]
                if np.sum(first_10 > 0.5):
                    # spike near beginning of trace, exceeds RF amplitude limits
                    try:
                        wf_adc_to_v[:10] = np.mean(first_10[first_10 < 0.5])
                    except:
                        print("Wide spike found at beginning of a {c} trace!".format(c=self.ch_names[chn]))
                        wf_adc_to_v[:10] = np.mean((wf_adc_to_v[10:20])[wf_adc_to_v[10:20] < 0.5])

                if np.sum(last_10 > 0.5):
                    # spike near end of trace, exceeds RF amplitude limits
                    try:
                        wf_adc_to_v[-10:] = np.mean(last_10[last_10 < 0.5])
                    except:
                        print("Wide spike found at end of a {c} trace!".format(c=self.ch_names[chn]))
                        wf_adc_to_v[-10:] = np.mean((wf_adc_to_v[-20:-10])[wf_adc_to_v[-20:-10] < 0.5])
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
            # self.ch_time_bins += (ref_ch_0cell - ch_0cell)
            # Testing
            self.ch_time_bins += (ref_ch_0cell - ch_0cell) - self.cable_delays[chn]
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
        return time_left_of_zeros - (v_signals_left_of_zeros/crossing_slopes), np.sign(crossing_slopes)

    def _t0_ref_points(self, time_calibrated_bins, voltage_calibrations, f=np.array([0.1, 0.2, 0.9]), thr=0.01):
        """Finds trigger time of each pulse for t0. Defined as a fraction f of the maximum pulse height.
        Method changed from crocker_signals. Returns trg points, amplitude (baseline corrected), and baseline."""
        t0_time_bins = time_calibrated_bins[self.ch_names["t0"]]
        polarity = -1  # cherenkov
        if self.det_type == "lfs_en":
            polarity = 1
        self.buffer[:] = polarity * voltage_calibrations[self.ch_names["t0"]]

        f_pts = f.size
        t0_ref_time = np.ones([f_pts, 5]) * -10
        t0_ref_voltage = np.ones(5) * -10
        t0_baseline = np.ones(5) * -10

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

            # above already checks for left or right edge, bottom deals with noise above thresholds
            if (n_pulse > 1) & (np.sum(np.abs(t0_time_bins[next_trigger_idx] - t0_ref_time[:n_pulse - 1]) < 35) > 0):
                find_pulses = False
                continue  # This guesses that triggering on noise between pulses, so stop

            if (next_max_value - baseline) < thr:  # no more peaks above threshold, time to stop
                find_pulses = False
                continue # effectively a break

            window_signal_voltage = self.buffer[mask_left_idx:mask_right_idx]
            window_signal_tbins = t0_time_bins[mask_left_idx:mask_right_idx]

            trgs_t, trg_max_v = rise_time_points(window_signal_tbins, window_signal_voltage, baseline, f=f)

            t0_ref_time[..., n_pulse - 1] = trgs_t  # python index by 0...
            t0_ref_voltage[n_pulse - 1] = trg_max_v
            t0_baseline[n_pulse - 1] = baseline

            self.buffer[mask_left_idx:mask_right_idx] = np.min(self.buffer)
            # print("n_pulse: ", n_pulse)
            n_pulse += 1

        return t0_ref_time[..., :n_pulse], t0_ref_voltage[..., :n_pulse], t0_baseline

    def _detector_trigger(self, time_calibrated_bins, voltage_calibrations, f=0.2):
        """cherenkov or lfs_en channel name"""
        # det_name = [key for key in self.ch_names.keys() if key not in ('rf', 'lfs', 't0')][0]
        if self.det_type == 'lfs_en':
            det_name = 'lfs'
        else:
            det_name = 'cherenkov'
        det_time_bins = time_calibrated_bins[self.ch_names[det_name]]
        self.buffer[:] = voltage_calibrations[self.ch_names[det_name]]
        # print("det name: ", det_name)
        baseline = 0
        bl_edge = 100
        if self.det_type == "cherenkov":
            baseline = np.mean(self.buffer[100:200])
            bl_edge = 100
        elif self.det_type == "lfs_en":
            baseline = np.mean(self.buffer[20:80])
            bl_edge = 80

        self.buffer[:bl_edge] = np.min(self.buffer)
        # trg_t, trg_v = linear_interpolate_trigger(det_time_bins, self.buffer, baseline, f=f)
        if self.det_type == "cherenkov":
            trg_t, trg_v = linear_interpolate_trigger(det_time_bins, self.buffer, baseline, f=f)
        else:
            trg_t, trg_v = leading_edge_trigger(det_time_bins, self.buffer, baseline, thr=0.1)
        return trg_t, trg_v

    def test_rf_t0_points(self):
        board = self.board_ids[0]
        channels = self.channels[board]
        voltage_calibrated = self.event_voltage_calibrate(board, channels)
        time_calibrated_bins = self.event_timing_calibrate(board, channels)

        # t0_frac = np.array([0.1, 0.2, 0.9])
        t0_frac = np.array([0.1, 0.2, 0.9])
        crossings, slope_sign = self._rf_ref_points(time_calibrated_bins, voltage_calibrated)
        t0_trigs, t0_max_pulse_voltages, t0_baselines = self._t0_ref_points(time_calibrated_bins, voltage_calibrated, f=t0_frac)
        print("t0_trgs: ", t0_trigs)
        det_trig, det_voltage_at_trig = self._detector_trigger(time_calibrated_bins, voltage_calibrated)

        fig, ax = plt.subplots(1, 1)
        for chn in channels:  # voltage and plotting
            polarity = 1
            if chn == self.ch_names["t0"]:
                polarity = -1
            ax.plot(time_calibrated_bins[chn], voltage_calibrated[chn] * polarity)
            # ax.plot(voltage_calibrated[chn] * polarity)
        ax.plot(crossings, np.zeros(crossings.size), "kX")
        # ax.plot(np.squeeze(t0_trigs), t0_voltage_at_trig, "o")  # for single trigger
        for f, trg in zip(t0_frac, t0_trigs):
            thr_voltages = t0_max_pulse_voltages * f
            baseline = t0_baselines
            ax.plot(trg, thr_voltages + baseline, "o")

        ax.plot(det_trig, det_voltage_at_trig, "8")
        ax.set_xlabel('time (ns)')
        ax.set_ylabel('amplitude (V)')
        plt.show()

    def t0_statistics(self, log_scale=False):
        """Generates histogram of 10-90 rise times and amplitudes of t0 signal."""
        board = self.board_ids[0]
        channels = self.channels[board]
        t0_frac = np.array([0.1, 0.9])

        rise_times, rt_bins = np.histogram([], bins=np.linspace(0.5, 3, num=1001))
        amps, amp_bins = np.histogram([], bins=np.linspace(0, 0.2, num=2001))

        rise_time_buffer = np.zeros(50000)  # temp storage
        rt_amp_buffer = np.zeros(50000)
        ptr = 0  # ptr to current point in buffer
        (evts, missed_evts) = (0, 0)  # below threshold

        keep_reading = True

        try:
            while keep_reading:
                voltage_calibrated = self.event_voltage_calibrate(board, channels)
                time_calibrated_bins = self.event_timing_calibrate(board, channels)

                # crossings, slope_sign = self._rf_ref_points(time_calibrated_bins, voltage_calibrated)
                t0_trigs, t0_max_voltages, _ = self._t0_ref_points(time_calibrated_bins,
                                                                                    voltage_calibrated, f=t0_frac)
                # det_trig, det_voltage_at_trig = self._detector_trigger(time_calibrated_bins, voltage_calibrated)

                event_rise_times = t0_trigs[1] - t0_trigs[0]  # second row (0.9) - first row (0.1)
                t0_pulses = event_rise_times.size
                if t0_pulses == 0:
                    print("No pulses found in an event")
                    continue
                rise_time_buffer[ptr:ptr+t0_pulses] = event_rise_times
                rt_amp_buffer[ptr:ptr+t0_pulses] = t0_max_voltages

                ptr += t0_pulses  # current index into buffer
                evts += t0_pulses  # total number of events
                missed_evts += (5 - t0_pulses)  # ideally 5 because of crocker RF period (44.4 ns)

                if (ptr + t0_pulses) > (rise_time_buffer.size - 20):  # Next set of events close to end of buffer
                    print("Appending to histograms. Full buffers.")
                    rise_times += np.histogram(rise_time_buffer[:ptr], bins=rt_bins)[0]
                    amps += np.histogram(rt_amp_buffer[:ptr], bins=amp_bins)[0]
                    ptr = 0  # back to beginning of buffer
                self.event = next(self.f)  # move to next event, stop iteration otherwise

        except StopIteration:
            print("Reached last event!")
            keep_reading = False
            pass
        finally:
            # empty buffers
            rise_times += np.histogram(rise_time_buffer[:ptr], bins=rt_bins)[0]
            amps += np.histogram(rt_amp_buffer[:ptr], bins=amp_bins)[0]

            print("Total pulses: ", evts)
            print("Missed pulses: ", missed_evts)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 12))  # ax1 -> rise_times, ax2 -> amplitudes
            fig.suptitle("T0 Rise Time and Max Pulse Voltage", fontsize=22)
            for ax, bin_edges, values, xlbl in zip((ax1, ax2), (rt_bins, amp_bins), (rise_times, amps), ("time (ns)", "max voltage (V)")):
                bins = (bin_edges[1:] + bin_edges[:-1])/2
                ax.step(bins, values, 'b-', where='mid')
                ax.set_xlabel(xlbl, fontsize=18)
                ax.set_ylabel("counts", fontsize=18)
                ax.tick_params(axis='both', labelsize=16)

                if log_scale:
                    ax.set_yscale('log')

            plt.show()

    def rf_to_t0_and_detector(self, log_scale=False):
        """Generates histograms of t0 - rf, detector - rf, and detector - t0"""
        # TODO: Detector - t0
        board = self.board_ids[0]
        channels = self.channels[board]
        t0_frac = np.array([0.2])  # "CFD" for t0

        t0_to_rf_times, t_bins = np.histogram([], bins=np.linspace(-44, 44, num=1001))
        det_to_rf_times, _ = np.histogram([], bins=t_bins)
        det_to_t0_times, _ = np.histogram([], bins=t_bins)

        t0_rf_time_buffer = np.zeros(50000)  # temp storage
        det_rf_time_buffer = np.zeros(50000)
        det_t0_time_buffer = np.zeros(50000)  # add

        ptr_g, ptr_rf = 0, 0  # ptr to current point in buffer for gamma and rf, respectively
        (evts, missed_evts) = (0, 0)  # below threshold

        keep_reading = True

        check = 1

        try:
            while keep_reading:
                voltage_calibrated = self.event_voltage_calibrate(board, channels)
                time_calibrated_bins = self.event_timing_calibrate(board, channels)

                crossings, slope_sign = self._rf_ref_points(time_calibrated_bins, voltage_calibrated)
                t0_trigs, t0_max_voltages, _ = self._t0_ref_points(time_calibrated_bins,
                                                                   voltage_calibrated, f=t0_frac)
                det_trig, det_voltage_at_trig = self._detector_trigger(time_calibrated_bins, voltage_calibrated)

                if check < 2:
                    print("det trig: ", det_trig)
                    print("t0_trigs: ", t0_trigs)
                    print("t0_trigs.shape: ", t0_trigs.shape)
                    print("t0_trigs[0]]: ", t0_trigs[0])
                    print("subtraction: ", det_trig - t0_trigs)
                    print("abs value of subtraction: ", np.abs(det_trig - t0_trigs[0]))
                    print("argmin: ", (np.abs(det_trig - t0_trigs[0])).argmin())
                    check += 1

                rf_ref_idx = (np.abs(det_trig - crossings)).argmin()
                t0_ref_idx = (np.abs(det_trig - t0_trigs[0])).argmin()
                # rf_to_t0 = t0_trigs[0][t0_ref_idx] - crossings[rf_ref_idx]
                tgamma_to_rf = (det_trig - crossings)[rf_ref_idx]  # gamma relative to closest RF
                tgamma_to_t0 = (det_trig - t0_trigs[0])[t0_ref_idx]  # + rf_to_t0

                if (tgamma_to_t0 > 2) & (self.det_type == "lfs_en"):
                    # Shift issue
                    self.event = next(self.f)
                    continue

                # TODO: fix need to have to use t0_trigs[0] instead of t0_trigs
                del_t = correlate_pulse_trains(t0_trigs, crossings[slope_sign < 0])
                # t0 relative to negative slope crossing RF
                t0refs = del_t.size  # how many t0-rf pairs exist

                det_rf_time_buffer[ptr_g:ptr_g + 1] = tgamma_to_rf
                det_t0_time_buffer[ptr_g:ptr_g + 1] = tgamma_to_t0
                t0_rf_time_buffer[ptr_rf:ptr_rf + t0refs] = del_t

                ptr_g += 1  # current index into  gamma buffer
                ptr_rf += t0refs

                evts += t0refs  # total number of events
                missed_evts += (5 - t0refs)  # ideally 5 because of crocker RF period (44.4 ns)

                if (ptr_rf + t0refs) > (t0_rf_time_buffer.size - 20):  # Next set of events close to end of buffer
                    print("Appending to rf-t0 histograms. Full rf-t0 buffers.")
                    t0_to_rf_times += np.histogram(t0_rf_time_buffer[:ptr_rf], bins=t_bins)[0]
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
            t0_to_rf_times += np.histogram(t0_rf_time_buffer[:ptr_rf], bins=t_bins)[0]
            det_to_rf_times += np.histogram(det_rf_time_buffer[:ptr_g], bins=t_bins)[0]
            det_to_t0_times += np.histogram(det_t0_time_buffer[:ptr_g], bins=t_bins)[0]

            print("Total pulses: ", evts)
            print("Missed pulses: ", missed_evts)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 12))  # ax1 -> rise_times, ax2 -> amplitudes
            if "cherenkov" in self.ch_names.keys():
                det = "Cherenkov"
            else:  # LFS
                det = "LFS"
            fig.suptitle(det + " Detector, RF, and T0 $\Delta$T", fontsize=22)

            for ax, bin_edges, values, \
                xlbl, title, plot_label in zip((ax1, ax2), (t_bins, t_bins),
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
            if self.det_type == "lfs_en":
                t_centers += 12.5
            else:
                t_centers += 7.5

            ax2.step(t_centers, det_to_t0_times, 'g-', where='mid', label="$T_{\gamma}-T_{t0}$")
            ax2.legend(loc='best')
            # TODO: ax.step(det to RF)

            plt.show()


def test_triggers(fname):
    tst = CrockerSignals(fname)
    print(tst.f.board_ids)

    skip = 508
    for _ in np.arange(skip):
        event = next(tst.f)
        # print(event.timestamp)

    event = next(tst.f)
    tst.test_rf_t0_points()


def t0_statistics(fname):
    t0data = CrockerSignals(fname)
    print(t0data.f.board_ids)
    t0data.t0_statistics()


def t0_rf_det_delta_t(fname, det):
    t0data = CrockerSignals(fname, det=det)
    print(t0data.f.board_ids)
    t0data.rf_to_t0_and_detector()


if __name__ == "__main__":
    import os
    from pathlib import Path

    # data_file_name = "20221017_Crocker_31.6V_cherenkov_500pa_DualDataset_nim_amp_p2_v10.dat"  # cherenkov
    # det = "cherenkov"
    data_file_name = "20221017_Crocker_31.6V_LFS_500pa_SingleDataset_nim_amp_p2_v19.dat"  # LFS
    det = "lfs_en"
    fname = os.path.join(str(Path(os.getcwd()).parents[1]), "sample_data", "drs4", data_file_name)

    # test_triggers(fname)
    t0_statistics(fname)
    # t0_rf_det_delta_t(fname, det)
