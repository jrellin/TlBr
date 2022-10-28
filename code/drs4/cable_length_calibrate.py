from binio import DRS4BinaryFile
import numpy as np
import matplotlib.pyplot as plt


def linear_interpolate_trigger(time_bins, waveform, baseline, f=0.2):
    """Assumes positive polarity signals"""
    wf = waveform - baseline
    t = time_bins
    max_sig_fv = f * np.max(wf)

    ati = np.argmax(np.sign(wf - max_sig_fv))  # (a)fter (t)rigger (i)ndex
    m = (wf[ati] - wf[ati-1]) / (t[ati]-t[ati-1])  # slope
    interp_t = t[ati-1] + ((max_sig_fv - wf[ati-1])/m)

    return interp_t, max_sig_fv + baseline  # add back baseline for plotting


class CableLengthCalibrate(object):

    def __init__(self, filename):
        self.f = DRS4BinaryFile(filename)
        # aliases
        self.n_boards = self.f.num_boards
        self.board = self.f.board_ids[0]
        self.time_widths = self.f.time_widths
        self.channels = self.f.channels
        self.n_channels = len(self.channels[self.board])
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
        cross_times = np.arange(self.n_channels)
        max_voltages = np.arange(self.n_channels)
        for chn in self.channels:
            min_v = np.min(wfs[chn])  # taken as baseline
            # max_v = np.max(wfs[chn])
            ct, mv = linear_interpolate_trigger(time_bins[chn], wfs[chn], min_v, f=f)
            cross_times[chn-1] = ct
            max_voltages[chn-1] = mv

        return cross_times, max_voltages

    def cable_calibration(self):
        """Plots rise times and max voltages of all channels. Provides statistics on mean and standard deviation."""
        board = self.board
        channels = self.channels[board]
        cross_f = np.array([0.2])  # "CFD", % of maximum

        rt_bins = np.linspace(10, 50, num=501)
        amp_bins = np.linspace(0, 0.5, num=2001)

        rise_times = {}
        amps = {}
        for chn in channels:
            rise_times[chn] = np.histogram([], bins=rt_bins)[0]
            amps[chn] = np.histogram([], bins=amp_bins)[0]

        bfr_size = 15000
        rise_time_buffer = np.zeros([self.n_channels, bfr_size])
        amp_buffer = np.zeros([self.n_channels, bfr_size])

        ptr = 0  # ptr to current point in buffer

        keep_reading = True

        check = 1
        # TODO: You are here


def calibrate_cables(fname):
    calib = CableLengthCalibrate(fname)
    print(calib.f.board_ids)


if __name__ == "__main__":
    import os
    from pathlib import Path

    data_file_name = "Cable_length_delay_calibration_crocker.dat"
    fname = os.path.join(str(Path(os.getcwd()).parents[1]), "sample_data", "drs4", data_file_name)

    calibrate_cables(fname)
