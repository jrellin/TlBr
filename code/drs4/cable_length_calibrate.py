from binio import DRS4BinaryFile
import numpy as np
import matplotlib.pyplot as plt


def linear_interpolate_trigger(time_bins, waveform, baseline, f=0.5):
    """Assumes positive polarity signals"""
    wf = waveform - baseline
    t = time_bins
    max_sig_fv = f * np.max(wf)

    ati = np.argmax(np.sign(wf - max_sig_fv))  # (a)fter (t)rigger (i)ndex
    m = (wf[ati] - wf[ati-1]) / (t[ati]-t[ati-1])  # slope
    if m != 0:
        interp_t = t[ati-1] + ((max_sig_fv - wf[ati-1])/m)
    else:
        interp_t = t[ati - 1]  # nothing to extrapolate

    return interp_t, max_sig_fv + baseline  # add back baseline for plotting


class CableLengthCalibrate(object):

    def __init__(self, filename):
        self.f = DRS4BinaryFile(filename)
        # aliases
        self.n_boards = self.f.num_boards
        self.board = self.f.board_ids[0]
        self.time_widths = self.f.time_widths
        self.channels = self.f.channels[self.board]
        self.n_channels = len(self.channels)
        self.event = next(self.f)  # first event
        self.ch_time_bins = np.zeros(1024)  # temporary working memory for time calibration data
        self.buffer = np.zeros(1024)  # temporary working memory for voltage calibration data
        self.ch_names = {"rf": 1, "cherenkov": 2, "lfs_timing": 3, "t0": 4}

    def event_voltage_calibrate(self, chns):
        voltage_calibrated = {}
        for chn in chns:
            voltage_calibrated[chn] = (self.event.adc_data[self.board][chn] / 65536) + \
                                      (self.event.range_center / 1000) - 0.5
        return voltage_calibrated

    def event_timing_calibrate(self, chns, ref=1):  # check ref is in chns before this gets called
        channels = np.array(chns)
        time_calibrated = {}

        trg_cell = self.event.trigger_cells[self.board]
        self.ch_time_bins[:] = np.cumsum(np.concatenate((np.array([0]),
                                                         self.time_widths[self.board][ref][trg_cell:1023],
                                                         self.time_widths[self.board][ref][:trg_cell])))
        time_calibrated[ref] = self.ch_time_bins.copy()
        ref_ch_0cell = self.ch_time_bins[(1024 - trg_cell) % 1024]

        for chn in channels[channels != ref]:
            self.ch_time_bins[:] = np.cumsum(np.concatenate((np.array([0]),
                                                             self.time_widths[self.board][chn][trg_cell:1023],
                                                             self.time_widths[self.board][chn][:trg_cell])))
            ch_0cell = self.ch_time_bins[(1024 - trg_cell) % 1024]
            self.ch_time_bins += (ref_ch_0cell - ch_0cell)
            time_calibrated[chn] = self.ch_time_bins.copy()

        return time_calibrated

    def fraction_amplitude_times(self, time_bins, wfs, f=0.5):
        cross_times = {}
        cross_voltages = {}
        for chn in self.channels:
            min_v = np.min(wfs[chn])  # taken as baseline
            # max_v = np.max(wfs[chn])
            ct, mv = linear_interpolate_trigger(time_bins[chn], wfs[chn], min_v, f=f)
            cross_times[chn] = ct
            cross_voltages[chn] = mv

        return cross_times, cross_voltages

    def cable_calibration(self, cross_f=0.5):
        """Plots rise times and max voltages of all channels. Provides statistics on mean and standard deviation."""
        board = self.board
        channels = self.channels

        rt_bins = np.linspace(75, 110, num=501)
        amp_bins = np.linspace(0, 0.2, num=1001)

        rise_times = {}
        amps = {}
        rise_time_buffer = {}
        amp_buffer = {}
        bfr_size = 15000
        for chn in channels:
            rise_times[chn] = np.histogram([], bins=rt_bins)[0]
            rise_time_buffer[chn] = np.zeros(bfr_size)
            amps[chn] = np.histogram([], bins=amp_bins)[0]
            amp_buffer[chn] = np.zeros(bfr_size)

        ptr = 0  # ptr to current point in buffer
        keep_reading = True

        check = 1
        # TODO: You are here

        try:
            while keep_reading:
                voltage_calibrated = self.event_voltage_calibrate(channels)
                time_calibrated_bins = self.event_timing_calibrate(channels)

                cross_times, cross_voltages = self.fraction_amplitude_times(time_calibrated_bins, voltage_calibrated, f=cross_f)

                for chn in channels:
                    rise_time_buffer[chn][ptr] = cross_times[chn]
                    amp_buffer[chn][ptr] = cross_voltages[chn]

                ptr += 1

                if ptr >= bfr_size:  # buffers full, histogram
                    print("Full buffers. Histogramming.")
                    for chn in channels:
                        rise_times[chn] = np.histogram(rise_time_buffer[chn], bins=rt_bins)[0]
                        amps[chn] = np.histogram(amp_buffer[chn], bins=amp_bins)[0]
                    ptr = 0

                self.event = next(self.f)
        except StopIteration:
            print("Reached last event!")
        finally:
            print("Emptying remaining buffers.")

            for chn in channels:
                rise_times[chn] = np.histogram(rise_time_buffer[chn][:ptr], bins=rt_bins)[0]
                amps[chn] = np.histogram(amp_buffer[chn][:ptr], bins=amp_bins)[0]

            rt_center_bins = 0.5 * (rt_bins[1:] + rt_bins[:-1])
            amp_center_bins = 0.5 * (amp_bins[1:] + amp_bins[:-1])
            rt_maximums = {}  # most encountered rise time value
            v_crossovers = {}  # most encountered crossover voltage

            for chn in channels:  # print these stats
                rt_maximums[chn] = rt_center_bins[np.argmax(rise_times[chn])]
                v_crossovers[chn] = amp_center_bins[np.argmax(amps[chn])]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 12))  # ax1 -> rise_times, ax2 -> amplitudes
            fig.suptitle("Cable Length Delays", fontsize=22)

            ch_names_by_ch = {v: k for k, v in self.ch_names.items()}  # keys are ch numbers, values are ch names
            # print("Ch names by channel: ", ch_names_by_ch)

            for ax, bin_centers, dict_values, xlbl in zip((ax1, ax2), (rt_center_bins, amp_center_bins), (rise_times, amps),
                                                    ("time (ns)", "max {f} voltage (V)".format(f=cross_f))):
                for chn in channels:
                    ax.step(bin_centers, dict_values[chn], where='mid', label=ch_names_by_ch[chn])
                ax.set_xlabel(xlbl, fontsize=18)
                ax.set_ylabel("counts", fontsize=18)
                ax.tick_params(axis='both', labelsize=16)

            ax1.legend(loc='best')

            print("=====Cable statistics=====")
            print("Channels: ", ch_names_by_ch)
            print("Rise Time Maximums: ", rt_maximums)
            print("Voltage Cross Over Maximums: ", v_crossovers)
            print("=========================")
            plt.show()


def calibrate_cables(fname):
    calib = CableLengthCalibrate(fname)
    print(calib.f.board_ids)
    calib.cable_calibration()


if __name__ == "__main__":
    import os
    from pathlib import Path

    data_file_name = "Cable_length_delay_calibration_crocker.dat"
    fname = os.path.join(str(Path(os.getcwd()).parents[1]), "sample_data", "drs4", data_file_name)

    calibrate_cables(fname)
