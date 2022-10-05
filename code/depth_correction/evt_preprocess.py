import numpy as np
import os


# Utility functions
# 9/13 - This code is to be used with DRS4 timing Data_hdf5 with the WaveDumpConfig_DRS4Triggering.txt config
def makedirs(path):
    """ Create directories for `path` (like 'mkdir -p'). """
    if not path:
        return
    folder = os.path.dirname(path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)


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


# Preprocess class for 88 inch cyclotron fields and Data_hdf5
class Preprocessor(object):
    """This class performs preprocessing steps on TlBr Data_hdf5"""
    read_block_size_limit = 20 * (2 ** 20)  # 20 mB in base 2

    def __init__(self, n_anodes=16, n_cathodes=1, verbose=False):
        self.n_anodes = n_anodes
        self.det_calibration = np.ones(self.n_anodes)  # set this otherwise, set to physical location
        self.n_cathodes = n_cathodes
        self.loaded_file = False
        self.f, self.fname = None, None  # file object, file name (as str)
        # self.ev_len, self.daq_dtypes = None, None
        self.ev_len = None
        self.max_block_read_limit = self.read_block_size_limit  # in bytes, will be overwritten
        self._daq_dtypes = None
        self.raw_samples = None
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
        # df = np.dtype(np.double).newbyteorder('<')

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
            self.raw_samples = np.frombuffer(self.f.read(2), dtype=np.dtype(np.uint16).newbyteorder('<'))[0]
            # assumes points field is first
        else:
            self.raw_samples = raw_samples  # can force it if first Data_hdf5 field is corrupted
        # self.ev_len, self.daq_dtypes = self._setup_data_fields()  # since raw_samples is now known
        self.daq_dtypes = self._setup_data_fields()  # since raw_samples is now known
        # self.max_block_read_limit = (self.read_block_size_limit // self.ev_len) * self.ev_len # since ev_len now known
        self.f.seek(0)

    def _check_buffer_integrity(self, buffer):
        """Accepts a batch buffer. Attempts to recover from missing Data_hdf5.
        Returns buffer of whole (no missing Data_hdf5) events and the remainder, if any, at end of the batch buffer."""

        samples = self.raw_samples
        bfr_even = ((len(buffer) % 2) ^ 1)
        remainder = b''
        new_buffer = b''

        arr = np.frombuffer(buffer[0:(len(buffer) - (bfr_even ^ 1))],
                            dtype=np.dtype(np.uint16).newbyteorder('<'))  #

        arr_shift_by_one = np.frombuffer(buffer[1:(len(buffer) - bfr_even)],
                                         dtype=np.dtype(np.uint16).newbyteorder('<'))

        combined_pts = np.concatenate(((2 * (np.argwhere(arr == samples))).flatten(),
                                       (2 * np.argwhere(arr_shift_by_one == samples)).flatten() + 1))

        if combined_pts.size == 0:  # no points field found
            print("Nothing found!")
            return b'', b''

        beg_event_locations = np.sort(combined_pts)  # These are point locations for found pts field for 8 byte Data_hdf5

        # if combined_pts.size > 0:
        arr_8byte = np.frombuffer(buffer, dtype=np.dtype(np.uint8).newbyteorder('<'))
        arr_chunks = np.split(arr_8byte, beg_event_locations)
        valid_evt_chunks = list(filter(lambda chk: chk.size == self.ev_len, arr_chunks))
        if len(valid_evt_chunks):
            print("Valid events found!")
            new_buffer = np.concatenate(valid_evt_chunks).tobytes()
            if arr_chunks[-1].size != self.ev_len:
                remainder = arr_chunks[-1].tobytes()
                print("arr_chunks[-1].size: ", arr_chunks[-1].size)
                # print("samples: ", samples)
                # print("Valid Events Arr Chunks: ", len(arr_chunks))
                # print("Combined Points: ", combined_pts.size)
                # print("Valid Events Remainder length: ", len(remainder))
        else:
            print("Arr chunks: ", len(arr_chunks))
            print("Combined Points: ", combined_pts.size)

        return new_buffer, remainder

    def _read_chunk(self):
        """Generator to read a chunk of Data_hdf5 from a large file.
         Returns buffered reads from file until there is no more"""
        while True:
            data = self.f.read(self.max_block_read_limit)
            if not data:
                break
            yield data

    def create_clean_file(self, output_fname=None):
        """This function converts a loaded binary file into a new output binary file with only events with all event
         field Data_hdf5. v2 attempts to limit memory access with a generator for reads instead of a while loop."""

        if not self.loaded_file:  # No file loaded
            print("FILE NOT LOADED! Nothing to process!")
            return
        # if output_fname is None:
        #     output_fname = os.path.splitext(os.path.basename(self.fname))[0] + '_clean.dat'

        if output_fname is None:
            output_fname = os.path.join(os.getcwd(), 'Data_processed_binary',
                                      os.path.splitext(os.path.basename(self.fname))[0] + '_clean.dat')
        makedirs(output_fname)

        with open(output_fname, 'wb') as outfile:
            total_possible_evts_read = 0
            total_good_events_read = 0
            self.f.seek(0)  # start at beginning of file, regardless of where you were
            remainder = b''  # This is the end of a batch buffer since you might cut off the last event between buffers

            for buffer in self._read_chunk():
                if len(buffer) < self.max_block_read_limit:  # EoF
                    print("Saw end of file!")
                    buffer = buffer[:-2]  # get rid of last EOF tag which is 2 bytes long (64999)

                possible_evts_read = len(buffer) // self.ev_len
                # this is the maximum new events that could be read from buffer

                total_possible_evts_read += possible_evts_read

                print("======================")
                print("Length of buffer in MB: ", len(buffer)/(1024*1024))
                print("Length of remainder in MB: ", len(remainder)/(1024*1024))
                print("======================")

                # curr_buffer = remainder + buffer[:possible_evts_read * self.ev_len]
                # Now to filter the buffer to remove incomplete Data_hdf5, adding back in any remainder of previous batch
                filtered_buffer, remainder = self._check_buffer_integrity(remainder +
                                                                          buffer[:(possible_evts_read * self.ev_len)])
                good_evts = len(filtered_buffer) // self.ev_len
                if good_evts == 0:
                    print("No good events!")
                    outfile.flush()
                    continue

                total_good_events_read += good_evts

                outfile.write(filtered_buffer)
                outfile.flush()

                if self.verbose:
                    print("Total events read so far: ", total_possible_evts_read)
                    print("Total good events read so far: ", total_good_events_read)
                    print("Percentage of good events in the Data_hdf5: ", total_good_events_read / total_possible_evts_read)
                    print("Output file size (bytes): ", outfile.tell())

    def batch_check_file_integrity(self, record_length=-1):
        """Goes through the entire loaded file and checks for location of points field.
        Used with check file integrity"""
        if not self.loaded_file:  # No file loaded
            print("FILE NOT LOADED! Nothing to return!")
            return

        samples = self.raw_samples
        if record_length > 0:
            samples = record_length

        if self.max_block_read_limit % 2:  # buffer size is odd number of bytes
            bfr_even = False
        else:  # buffer size is even number of bytes
            bfr_even = True

        self.f.seek(0)  # start at beginning of file, regardless of where you were
        last_batch_evt_location = -self.ev_len

        while True:
            buffer = self.f.read(self.max_block_read_limit)
            if not buffer:  # nothing more to read
                break

            if len(buffer) < self.max_block_read_limit:  # EoF
                print("Saw end of file!")
                buffer = buffer[:-2]  # get rid of last EOF tag which is 2 bytes long (64999)

            arr = np.frombuffer(buffer[0:(len(buffer) - (bfr_even ^ 1))],
                                dtype=np.dtype(np.uint16).newbyteorder('<'))  #

            arr_shift_by_one = np.frombuffer(buffer[1:(len(buffer) - bfr_even)],
                                             dtype=np.dtype(np.uint16).newbyteorder('<'))

            combined_pts = np.concatenate(((2 * (np.argwhere(arr == samples))).flatten(),
                                           (2 * np.argwhere(arr_shift_by_one == samples)).flatten() + 1))
            if combined_pts.size == 0:
                beg_event_locations = np.r_[last_batch_evt_location, np.sort(combined_pts)]

                evt_deltas = beg_event_locations[1:] - beg_event_locations[:-1]  # distance in bytes between
                last_batch_evt_location = beg_event_locations[-1] - len(buffer)

                yield evt_deltas, (len(buffer) // self.ev_len)  # , \
            else:
                yield 0, 0


def check_file_integrity(file_name, n_anodes=16, det_calib=None, verbose=False, record_length=-1):
    """Spits out histogram of event size based on location of points field.
    Record length comes from file if set to -1 """
    import matplotlib.pyplot as plt

    prs = Preprocessor(n_anodes=n_anodes)  # n_cathode = 1
    if det_calib is not None:
        prs.det_calibration = det_calib
    prs.load_binary_file(file_name)

    ev_len = prs.ev_len

    ch_names = []
    for n in np.arange(1, n_anodes + 1):
        ch_names.append('ch' + str(n))

    hist_delta_bytes, hist_edges = np.histogram([], range=(0, 3 * ev_len), bins=(3 * ev_len // 4))
    bin_centers = (hist_edges[:-1] + hist_edges[1:])/2

    evts_read_total = 0
    evts_found = 0  # i.e. found out field
    gen = 0

    for byte_deltas, tot_evts in prs.batch_check_file_integrity(record_length=record_length):
        evts_read_total += tot_evts
        evts_found += byte_deltas.size
        hist_delta_bytes += np.histogram(byte_deltas, bins=hist_edges)[0]
        if verbose:
            print("Batch number: ", gen)
            print("Events Read: ", tot_evts)
            print("Events Found: ", byte_deltas.size)
            print("Next batch!")
            print("")
        gen += 1

    print("Total events: ", evts_read_total)
    print("Number of Points Field Occurrences: ", evts_found)
    print("Fraction of Points Field found: ", evts_found/evts_read_total)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.set_xlabel('Byte Deltas')
    ax.set_ylabel('Counts')
    ax.set_yscale('log')
    ax.set_title("Byte Deltas Between Points Fields")
    ax.step(bin_centers, hist_delta_bytes, 'b-', where='mid')
    plt.show()


def create_clean_file(file_name, n_anodes=16, verbose=False, **kwargs):
    # kwargs -> output_fname=None
    prs = Preprocessor(n_anodes=n_anodes, verbose=verbose)  # n_cathode = 1
    prs.load_binary_file(file_name)
    prs.create_clean_file(**kwargs)


def main():
    base_folder = "C:/Users/tlbr-user/Documents/TlBr_daq_analysis_wc/LocalDigitizerData/"
    # data_file_name = "DavisD2022_9_16T11_18.dat"
    # data_file_name = "DavisD2022_9_16T16_10.dat"  # BIG Data set, CG 32, FG 0, drs4 trigger
    data_file_name = "DavisD2022_9_20T17_1.dat"  # 200k events, overnight, coarse gain = 16, fine gain 127


    fname = base_folder + data_file_name
    n_anodes = 16
    # check_file_integrity(fname,n_anodes=n_anodes, debug=False)
    create_clean_file(fname, n_anodes=n_anodes, verbose=True)


if __name__ == "__main__":
    main()
