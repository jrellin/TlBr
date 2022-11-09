import numpy as np
import os
from binio import DRS4BinaryFile
# This file is incomplete, meant to parse binary cci and binary sipm


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


# Parser that combines data sets into one hdf5 file
class Combiner(object):
    def __init__(self, n_anodes=16, n_cathodes=1, verbose=False):
        self.n_anodes = n_anodes
        self.det_calibration = np.ones(self.n_anodes)  # set this otherwise, set to physical location
        self.n_cathodes = n_cathodes
        # self.loaded_file = False
        self.f, self.fname = None, None  # file object, file name as string
        self.ev_len = None
        self._cci_dtypes = None
        self.raw_samples = None  # TODO
        self.pin_map = np.array([[2, 1, 3, 4], [15, 8, 6, 5], [13, 11, 9, 7], [16, 14, 12, 10]])
        # channel map as pins as plugged into digitizer
        self.det_map = np.arange(1, 16+1).reshape([4, 4])
        # channel map as located in space, "top left" is channel 1, 16 is "bottom right" with TlBr
        # in upper left corner of pixel board
        self.cci_binary_loaded = False
        self.sipm_binary_loaded = False
        self.verbose = verbose
        self.cci_file_ptr = 0  # first event starts at 0 in seek events, TODO: Model like binio
        self.sipm_file = None

    @property
    def cci_dtypes(self):
        return self._cci_dtypes

    @cci_dtypes.setter
    def cci_dtypes(self, val):
        self._cci_dtypes = val
        if val is None:
            self.ev_len = 0
        else:
            self.ev_len = val.itemsize

    def _setup_cci_data_fields(self):
        ds16 = np.dtype(np.int16).newbyteorder('<')  # signed 16 integer
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

        shaped_cathode_field = ['shaped_cathode_max']
        shaped_cathode_dtype = [d16]
        data_fields.extend(shaped_cathode_field)
        digitizer_types.extend(shaped_cathode_dtype)

        trgs_field = ['anode_trgs']
        trgs_dtype = [(ds16, (10,))]
        data_fields.extend(trgs_field)
        digitizer_types.extend(trgs_dtype)

        trgs_cathode_field = ['cathode_trgs']
        trgs_cathode_dtype = [(ds16, (3,))]
        data_fields.extend(trgs_cathode_field)
        digitizer_types.extend(trgs_cathode_dtype)

        daq_dtypes = np.dtype({"names": data_fields, "formats": digitizer_types})

        # ev_len = daq_dtypes.itemsize  # in bytes

        if self.verbose:
            print("raw_samples: ", self.raw_samples)
            print("data_fields: ", data_fields)
            print("digitizer dtypes : ", digitizer_types)
            # print("ev_len (in bytes): ", ev_len)
        # return ev_len, daq_dtypes
        return daq_dtypes

    def load_cci_binary_file(self, fname, raw_samples=-1):
        self.fname = fname
        self.f = _load_data_file(fname)
        self.cci_binary_loaded = True
        if raw_samples < 0:
            self.raw_samples = np.frombuffer(self.f.read(2), dtype=np.dtype(np.uint16).newbyteorder('<'))[0]  # assumes points field is first
        else:
            self.raw_samples = raw_samples  # can force it if first Data_hdf5 field is corrupted
        self.cci_dtypes = self._setup_cci_data_fields()  # since raw_samples is now known
        # self.max_block_read_limit = (self.read_block_size_limit // self.ev_len) * self.ev_len # since ev_len now known
        self.f.seek(0)

    def load_drs4_binary_file(self, drs4_fname, ch=4):
        if ch not in (1, 4):
            ValueError('{ch} must be either 1 (button source measurmeents) or 4 (Crocker).'.format(ch=ch))
        if ch is 4:
            from sipm_4ch_drs4 import SiPM4
            self.sipm_file = SiPM4(drs4_fname)
        else:
            from sipm_1ch_drs4 import SiPM1
            self.sipm_file = SiPM1(drs4_fname)
        self.sipm_binary_loaded = True

    def seek_events(self, evt=0,  return_to_beginning=False):
        """Read an event starting at event number evt (starts at 0)."""
        self.f.seek(self.ev_len * evt)
        arr = np.frombuffer(self.f.read(self.ev_len), dtype=self.cci_dtypes)
        # arr_dict = self._parsed_array_to_dict(arr)
        if return_to_beginning:
            self.f.seek(0)  # return to beginning of file
        return arr


def main():
    pass


if __name__ == "__main__":
    main()

