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
        self.ch_time_bins = np.zeros(1024)  # temporary working memory
        # self.ch_names = ["rf", "lfs", "cherenkov", "t0"]
        self.ch_names = {"rf": 1, "lfs": 2, "cherenkov": 3, "t0": 4}

    # def event_voltage_calibrate(self):
    #    for b in self.board_ids:
    #        for chn in self.channels[b]:
    #            yield (self.event.adc_data[b][chn] / 65536) + (self.event.range_center/1000) - 0.5

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

    def _t0_ref_points(self, time_calibrated_bins, voltage_calibrations, f=0.2):
        """Finds rise time of each pulse for t0. Defined as a fraction f of the maximum pulse height."""
        t0_waveform = -voltage_calibrations[self.ch_names["t0"]]  # flip to find peaks not troughs
        t0_time_bins = time_calibrated_bins[self.ch_names["t0"]]
        pass

    def test_rf_t0_points(self):
        board = self.board_ids[0]
        channels = self.channels[board]
        voltage_calibrated = self.event_voltage_calibrate(board, channels)
        time_calibrated_bins = self.event_timing_calibrate(board, channels)

        crossings, slope_sign = self._rf_ref_points(time_calibrated_bins, voltage_calibrated)
        fig, ax = plt.subplots(1, 1)
        for chn in channels:  # voltage and plotting
            ax.plot(time_calibrated_bins[chn], voltage_calibrated[chn])
            # ax.plot(corrected_voltage)
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

    for _ in np.arange(5):
        event = next(tst.f)
        print(event.timestamp)

    event = next(tst.f)
    tst.test_rf_t0_points()


if __name__ == "__main__":
    main()
