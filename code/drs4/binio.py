import numpy as np
import struct
from collections import namedtuple
from io import BufferedReader, FileIO
from datetime import datetime


Event = namedtuple(
    'Event',
    [
        'event_id',
        'timestamp',
        'range_center',
        'adc_data',
        'scalers',
        'trigger_cells',
    ]
)


class DRS4BinaryFile(BufferedReader):

    def __init__(self, filename):
        super().__init__(FileIO(filename, 'rb'))

        assert self.read(4) == b'DRS2', 'File does not seem to be a DRS4 binary file'
        assert self.read(4) == b'TIME', 'File does not contain TIME header'

        self.board_ids = []
        self.time_widths = {}
        self.channels = {}

        header = self.read(4)
        while header.startswith(b'B#'):
            board_id, = struct.unpack('H', header[2:])
            self.board_ids.append(board_id)
            self.time_widths[board_id] = {}
            self.channels[board_id] = []

            header = self.read(4)
            while header.startswith(b'C'):
                channel = int(header[1:].decode())
                self.channels[board_id].append(channel)

                self.time_widths[board_id][channel] = self._read_timewidth_array()

                header = self.read(4)

        self.num_boards = len(self.board_ids)

        self.seek(-4, 1)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            header = self.read(4)
        except IOError:
            raise StopIteration

        if header != b'EHDR':
            raise StopIteration

        event_id, = struct.unpack('I', self.read(4))
        year, month, day, hour, minute, second, ms = struct.unpack(
            '7H', self.read(struct.calcsize('7H'))
        )
        timestamp = datetime(year, month, day, hour, minute, second, ms * 1000)
        range_center, = struct.unpack('H', self.read(2))

        scalers = {}
        trigger_cells = {}
        adc_data = {}

        for board_id, channels in self.channels.items():
            assert self.read(2) == b'B#'
            assert struct.unpack('H', self.read(2))[0] == board_id

            assert self.read(2) == b'T#'
            trigger_cells[board_id], = struct.unpack('H', self.read(2))

            scalers[board_id] = {}
            adc_data[board_id] = {}

            for channel in channels:
                assert self.read(4) == 'C{:03d}'.format(channel).encode('ascii')

                scalers[board_id][channel], = struct.unpack('I', self.read(4))
                adc_data[board_id][channel] = self._read_adc_data()

        return Event(
            event_id=event_id,
            timestamp=timestamp,
            range_center=range_center,
            adc_data=adc_data,
            scalers=scalers,
            trigger_cells=trigger_cells,
        )

    def _read_timewidth_array(self):
        return np.frombuffer(self.read(1024 * 4), 'float32')

    def _read_adc_data(self):
        return np.frombuffer(self.read(1024 * 2), 'uint16')


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # import DRS4BinaryFile
    import os
    from pathlib import Path

    # data_file_name = "20221017_Crocker_31.6V_cherenkov_500pa_DualDataset_nim_amp_p2_v10.dat"
    data_file_name = "20221017_Crocker_31.6V_LFS_500pa_SingleDataset_nim_amp_p2_v19.dat"  # LFS
    # LFS -> RF, lfs (timing), lfs_en, t0
    fname = os.path.join(str(Path(os.getcwd()).parents[1]), "sample_data", "drs4", data_file_name)
    # print("fname: ", fname)

    with DRS4BinaryFile(fname) as f:
        print(f.board_ids)
        print(f.channels)

        skip = 4000  # 10000, 1004 # 4000 for LFS spike artifact
        for _ in np.arange(skip):
            next(f)

        # i = 0  # Below is how you can iterate on all events
        # try:
        #     for evt in f:
        #         i += 1
        # except StopIteration as e:
        #     pass
        # finally:
        #     print(i)

        event = next(f)
        print(event.event_id)
        print(event.timestamp)

        b0 = f.board_ids[0]  # first board
        print(event.scalers[b0])

        chs = len(event.adc_data[b0])
        trg_cell = event.trigger_cells[b0]
        corrected_voltage = np.zeros(1024)

        time_corrected = [np.zeros(1024)] * 4
        ref_ch0_cell = 0
        ch_time_bins = np.zeros(1024)

        for i in np.arange(chs):  # timing calibration, TODO: Allow for any channel to be reference
            ch_time_bins[:] = np.cumsum(np.concatenate((np.array([0]), f.time_widths[b0][i + 1][trg_cell:1023],
                                               f.time_widths[b0][i + 1][:trg_cell])))

            ch0_cell = ch_time_bins[(1024 - trg_cell) % 1024]
            if i == 0:  # first channel cell 0 is used as the global reference
                ref_ch0_cell = ch0_cell
                offset = 0
            else:
                offset = ref_ch0_cell - ch0_cell

            time_corrected[i][:] = ch_time_bins + offset

        fig, ax = plt.subplots(1, 1)
        for i in np.arange(chs):  # voltage and plotting
            corrected_voltage[:] = (event.adc_data[b0][i + 1]/65536) + (event.range_center/1000.0) - 0.5
            # plt.plot(event.adc_data[b0][i + 1])  # uncorrected
            # ax.plot(time_corrected[i], corrected_voltage)
            ax.plot(corrected_voltage)
        ax.set_xlabel('time (ns)')
        ax.set_ylabel('amplitude (V)')
        plt.show()

