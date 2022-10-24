from binio import DRS4BinaryFile
import numpy as np


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
        self.ch_names = ["rf", "lfs", "cherenkov", "t0"]

    # def event_voltage_calibrate(self):
    #    for b in self.board_ids:
    #        for chn in self.channels[b]:
    #            yield (self.event.adc_data[b][chn] / 65536) + (self.event.range_center/1000) - 0.5

    def event_voltage_calibrate(self, board, chn):
        return (self.event.adc_data[board][chn] / 65536) + (self.event.range_center / 1000) - 0.5

    def event_timing_calibrate(self, board, chns, ref=1):  # check ref is in chns before this gets called
        time_calibrated = {}

        trg_cell = self.event.trigger_cells[board]
        self.ch_time_bins[:] = np.cumsum(np.concatenate((np.array([0]), self.time_widths[board][ref][trg_cell:1023],
                                                         self.time_widths[board][ref][:trg_cell])))
        ref_ch_0cell = self.ch_time_bins[(1024 - trg_cell) % 1024]
        time_calibrated[ref] = self.ch_time_bins.copy()

        for chn in chns[chns != ref]:
            self.ch_time_bins[:] = np.cumsum(np.concatenate((np.array([0]), self.time_widths[board][chn][trg_cell:1023],
                                                             self.time_widths[board][chn][:trg_cell])))
            ch_0cell = self.ch_time_bins[(1024 - trg_cell) % 1024]
            self.ch_time_bins += (ref_ch_0cell - ch_0cell)
            time_calibrated[chn] = self.ch_time_bins.copy()

        return time_calibrated

    def rf_ref_points(self):
        pass


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


if __name__ == "__main__":
    main()
