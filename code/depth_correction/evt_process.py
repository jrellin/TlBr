import numpy as np
import os

# This is an event by event parser. Slower but safer than batch processing when missing bytes are suspected.
# 8/10/22 - 88 inch cyclotron
# 8/11/22 - actually at 88 inch cyclotron
# 9/13 - This code is to be used with DRS4 timing Data_hdf5 with the WaveDumpConfig_DRS4Triggering.txt config


# Utility functions
def _load_data_file(fname):
    if fname is None:
        ValueError('You must specify a filename to load a config file!')
    if isinstance(fname, str):  # Python3: basestring -> str
        if os.path.isfile(fname):
            try:
                return open(fname, 'rb')  # , os.path.getsize(fname)  # file object, size in bytes of file
            except Exception as e:
                print(e)
        else:
            raise ValueError('{fi} is not a path to a file'.format(fi=fname))
    else:
        raise ValueError('{fi} is not a string. It is a {ty}'.format(fi=fname, ty=type(fname).__name__))


def as_dict(rec):
    """ turn a numpy recarray record into a dict. """
    return {name: rec[name] for name in rec.dtype.names}


# Parser class for 88 inch cyclotron fields and Data_hdf5
class Parser(object):
    """This class performs on the fly parsing of Data_hdf5 from SIS3316 card(s) based on set config settings"""
    read_block_size_limit = 20 * (2 ** 20)  # 20 mB in base 2

    def __init__(self, n_anodes=16, n_cathodes=1, verbose=False):
        self.n_anodes = n_anodes
        self.det_calibration = np.ones(self.n_anodes)  # set this otherwise, set to physical location
        self.n_cathodes = n_cathodes
        self.loaded_file = False
        self.f, self.fname = None, None  # file object, file name as string
        self.ev_len = None
        self.max_block_read_limit = self.read_block_size_limit  # in bytes, will be overwritten
        self._daq_dtypes = None
        self.raw_samples = None  # TODO
        self.pin_map = np.array([[2, 1, 3, 4], [15, 8, 6, 5], [13, 11, 9, 7], [16, 14, 12, 10]])
        # channel map as pins as plugged into digitizer
        self.det_map = np.arange(1, 16+1).reshape([4, 4])
        # channel map as located in space, "top left" is channel 1, 16 is "bottom right" with TlBr
        # in upper left corner of pixel board
        self.verbose = verbose

    @property
    def daq_dtypes(self):
        return self._daq_dtypes

    @daq_dtypes.setter
    def daq_dtypes(self, val):
        self._daq_dtypes = val
        if val is None:
            self.ev_len = 0
        else:
            self.ev_len = val.itemsize
            self.max_block_read_limit = (self.read_block_size_limit // self.ev_len) * self.ev_len  # since ev_len now known

    def _setup_data_fields(self):
        d16 = np.dtype(np.uint16).newbyteorder('<')
        d32 = np.dtype(np.uint32).newbyteorder('<')
        d64 = np.dtype(np.uint64).newbyteorder('<')
        df = np.dtype(np.double).newbyteorder('<')

        data_fields = ['points', 'total_time', 'timestamp', 'event_counter']  # event fields
        digitizer_types = [d16, d64, d64, d32]  # Data_hdf5 types

        amp_fields = []
        bl_fields = []
        for idx in np.arange(1, self.n_anodes + 1):
            amp_fields.append('amp' + str(idx))
            bl_fields.append('bl' + str(idx))

        data_fields.extend(amp_fields)
        digitizer_types.extend([d16] * self.n_anodes)

        data_fields.extend(bl_fields)
        digitizer_types.extend([d16] * self.n_anodes)

        raw_cathode_channel_field = ['cathode_trace']  # 88 inch
        raw_cathode_channel_dtype = [(d16, (self.raw_samples,))]  # 88 inch
        data_fields.extend(raw_cathode_channel_field)  # 88inch
        digitizer_types.extend(raw_cathode_channel_dtype)  # 88 inch

        shaped_cathode_field = ['shaped_cathode_max']  # 88 inch
        shaped_cathode_dtype = [d16]  # 88 inch
        data_fields.extend(shaped_cathode_field)  # 88inch
        digitizer_types.extend(shaped_cathode_dtype)  # 88 inch

        raw_anode_timing_channel_field = ['timing_trace']  # 9/13 - DRS4 triggering
        raw_anode_timing_channel_dtype = [(d16, (self.raw_samples,))]  # 9/13 - DRS4 triggering
        data_fields.extend(raw_anode_timing_channel_field)  # 9/13 - DRS4 triggering
        digitizer_types.extend(raw_anode_timing_channel_dtype)  # 9/13 - DRS4 triggering

        cfd_field = ['last_timing']  # 9/13 - DRS4 triggering, index position of last anode trigger signal
        cfd_dtype = [d16]  # 9/13 - DRS4 triggering
        data_fields.extend(cfd_field)  # 9/13 - DRS4 triggering
        digitizer_types.extend(cfd_dtype)  # 9/13 - DRS4 triggering

        daq_dtypes = np.dtype({"names": data_fields, "formats": digitizer_types})

        # ev_len = daq_dtypes.itemsize  # in bytes

        if self.verbose:
            print("raw_samples: ", self.raw_samples)
            print("data_fields: ", data_fields)
            print("digitizer dtypes : ", digitizer_types)
            # print("ev_len (in bytes): ", ev_len)
        # return ev_len, daq_dtypes
        return daq_dtypes

    def load_binary_file(self, fname, raw_samples=-1):
        self.fname = fname
        self.f = _load_data_file(fname)
        self.loaded_file = True
        if raw_samples < 0:
            self.raw_samples = np.frombuffer(self.f.read(2), dtype=np.dtype(np.uint16).newbyteorder('<'))[0]  # assumes points field is first
        else:
            self.raw_samples = raw_samples  # can force it if first Data_hdf5 field is corrupted
        self.daq_dtypes = self._setup_data_fields()  # since raw_samples is now known
        # self.max_block_read_limit = (self.read_block_size_limit // self.ev_len) * self.ev_len # since ev_len now known
        self.f.seek(0)

    def _parsed_array_to_dict(self, arr, evts, filter_channel_max=False):
        """Converts a parsed array from buffer into a dictionary. Private method for batch_parse and seek_events."""
        tmp_array = np.zeros([self.n_anodes, evts])

        arr_dict = {'evts': evts}
        arr_dict['ch_timing'] = [-1] * self.n_anodes

        for n in np.arange(1, self.n_anodes + 1):
            det_ch = self.det_map[self.pin_map == n].item()  # maps digitizer channel to physical location
            # corrected = self.det_calibration[det_ch - 1] * (arr['amp' + str(n)] - arr['bl' + str(n)]).astype('float')
            corrected = self.det_calibration[det_ch - 1] * (arr['amp' + str(n)])  # no baseline subtract
            if not filter_channel_max:  # keep all event values for all channels
                arr_dict['ch' + str(det_ch)] = corrected
            tmp_array[det_ch - 1, :] = corrected

        arr_dict['max_anode'] = np.max(tmp_array, axis=0)

        if filter_channel_max:  # keep events for channel only if they have the largest amplitude of all channels
            for n in np.arange(1, self.n_anodes + 1):  # n is channel id
                idx = n - 1  # python indexes at 0
                ch_max_vals = np.argwhere(np.argmax(tmp_array, axis=0) == idx)  # i.e. where ch is max
                arr_dict['ch' + str(n)] = tmp_array[idx, ch_max_vals].flatten()
                arr_dict['ch_timing'][idx] = arr['last_timing'][ch_max_vals]

        # arr_dict['timing_channel'] = arr['timing_channel']  # 88 inch,
        arr_dict['cathode_trace'] = arr['cathode_trace']  # 88 inch
        arr_dict['shaped_cathode_max'] = arr['shaped_cathode_max']  # 88 inch
        arr_dict['timing_trace'] = arr['timing_trace']  # DRS4 Timing, 9/13
        arr_dict['last_timing'] = arr['last_timing']  # DRS4 Timing, 9/13, ALL time events

        arr_dict['points'] = arr['points'][0]  # Should all be the same
        arr_dict['timestamp'] = arr['timestamp']
        arr_dict['event_counter'] = arr['event_counter']
        return arr_dict

    def batch_parse(self, max_evt_read=-1, filter_channel_max=False, return_dict=True):
        """Read raw binary Data_hdf5 in blocks and output dict. Returns lazy iterator."""
        if not self.loaded_file:  # No file loaded
            print("FILE NOT LOADED! Nothing to return!")
            return

        evts_read = 0
        keep_reading = True  # useful for only reading part of the file based on max_evt_read
        self.f.seek(0)  # start at beginning of file, regardless of where you were

        while keep_reading:
            if self.verbose:
                print("Current position in file (in bytes): ", self.f.tell())
            buffer = self.f.read(self.max_block_read_limit)
            if not buffer:  # nothing more to read
                break

            if len(buffer) < self.max_block_read_limit:  # EoF
                print("Saw end of file!")
                buffer = buffer[:-2]  # get rid of last EOF tag which is 2 bytes long (64999)

            evts = len(buffer) // self.ev_len

            if (max_evt_read > 0) & ((evts_read + evts) > max_evt_read):
                print("keep_reading false loop!")
                evts = max_evt_read - evts_read  # remaining evts to limit
                keep_reading = False

            evts_read += evts

            arr = np.frombuffer(buffer[:(evts * self.ev_len)], dtype=self.daq_dtypes)
            if return_dict:  # This is really an event dictionary
                parsed_data = self._parsed_array_to_dict(arr, evts, filter_channel_max=filter_channel_max)
            else:
                parsed_data = arr

            if self.verbose:
                print("Total events read so far: ", evts_read)
            # print("arr_dict keys: ", arr_dict.keys())
            # arr_dict fields -> evts, ch1, ch2,....ch16, cathode_trace, max_anode, points, timestamp, event_counter
            yield parsed_data

    def seek_events(self, start=0, n=10):
        """Seek only a small block of n consecutive events starting at event number start."""
        self.f.seek(self.ev_len * start)
        arr = np.frombuffer(self.f.read(self.ev_len * n), dtype=self.daq_dtypes)
        # print("array points field: ", arr['points'])
        # print("arr (first 5 rows): ", arr)
        arr_dict = self._parsed_array_to_dict(arr, n)
        self.f.seek(0)  # return to beginning of file
        return arr_dict


def main():
    pass


if __name__ == "__main__":
    main()

