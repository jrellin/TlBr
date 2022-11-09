import numpy as np
import evt_process as evt_parser88
import matplotlib.pyplot as plt
import tables
import os


# Helper Functions
def makedirs(path):
    """ Create directories for `path` (like 'mkdir -p'). """
    if not path:
        return
    folder = os.path.dirname(path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)


def load_h5file(filepath):
    if tables.is_hdf5_file(filepath):
        h5file = tables.open_file(filepath, 'r')
        return h5file
    else:
        raise ValueError('{fi} is not a hdf5 file!'.format(fi=filepath))


def create_new_daq_dtype(samples, verbose=False):
    """This creates the Data_hdf5 types for particular WaveDump file. Edit this here to use evt_process."""
    n_anodes = 16
    ds16 = np.dtype(np.int16).newbyteorder('<')  # signed 16 integer
    d16 = np.dtype(np.uint16).newbyteorder('<')
    d32 = np.dtype(np.uint32).newbyteorder('<')
    d64 = np.dtype(np.uint64).newbyteorder('<')
    df = np.dtype(np.double).newbyteorder('<')

    data_fields = ['points', 'total_time', 'timestamp', 'event_counter']  # event fields
    digitizer_types = [d16, d64, d64, d32]  # Data_hdf5 types

    amp_fields = []
    bl_fields = []
    for idx in np.arange(1, n_anodes + 1):
        amp_fields.append('amp' + str(idx))
        bl_fields.append('bl' + str(idx))

    data_fields.extend(amp_fields)
    digitizer_types.extend([d16] * n_anodes)

    data_fields.extend(bl_fields)
    digitizer_types.extend([d16] * n_anodes)

    # raw_cathode_channel_field = ['cathode_trace']
    # raw_cathode_channel_dtype = [(d16, (samples,))]
    # data_fields.extend(raw_cathode_channel_field)
    # digitizer_types.extend(raw_cathode_channel_dtype)

    shaped_cathode_field = ['shaped_cathode_max']
    shaped_cathode_dtype = [d16]
    data_fields.extend(shaped_cathode_field)
    digitizer_types.extend(shaped_cathode_dtype)

    # shaped_sipm_field = ['shaped_sipm_max']  # 88 inch
    # shaped_sipm_dtype = [d16]  # 88 inch
    # data_fields.extend(shaped_sipm_field)  # 88inch
    # digitizer_types.extend(shaped_sipm_dtype)  # 88 inch

    trgs_field = ['anode_trgs']
    trgs_dtype = [(ds16, (10,))]
    data_fields.extend(trgs_field)
    digitizer_types.extend(trgs_dtype)

    trgs_cathode_field = ['cathode_trgs']
    trgs_cathode_dtype = [(ds16, (3,))]
    data_fields.extend(trgs_cathode_field)
    digitizer_types.extend(trgs_cathode_dtype)

    # raw_anode_timing_channel_field = ['timing_trace']  # 9/13 - DRS4 triggering
    # raw_anode_timing_channel_dtype = [(d16, (samples,))]  # 9/13 - DRS4 triggering
    # data_fields.extend(raw_anode_timing_channel_field)  # 9/13 - DRS4 triggering
    # digitizer_types.extend(raw_anode_timing_channel_dtype)  # 9/13 - DRS4 triggering

    # cfd_field = ['last_timing']  # 9/13 - DRS4 triggering, index position of last anode trigger signal
    # cfd_dtype = [d16]  # 9/13 - DRS4 triggering
    # data_fields.extend(cfd_field)  # 9/13 - DRS4 triggering
    # digitizer_types.extend(cfd_dtype)  # 9/13 - DRS4 triggering

    daq_dtypes = np.dtype({"names": data_fields, "formats": digitizer_types})

    ev_len = daq_dtypes.itemsize  # in bytes

    if verbose:
        print("----------New DAQ Dtypes----------")
        print("data_fields: ", data_fields)
        print("digitizer dtypes : ", digitizer_types)
        print("ev_len (in bytes): ", ev_len)
    return daq_dtypes


# Main Functions
def preprocess_binary_file(binary_file_name, n_anodes=16, verbose=False, **kwargs):
    """Process binary file to new binary file ensuring that no Data_hdf5 is missing"""
    # kwargs -> output_fname=None
    from evt_preprocess import Preprocessor
    prs = Preprocessor(n_anodes=n_anodes, verbose=verbose)
    prs.load_binary_file(binary_file_name)
    samples = prs.raw_samples
    prs.daq_dtypes = create_new_daq_dtype(samples, **kwargs)
    print("Raw samples: ", prs.raw_samples)
    print("daq_dtypes: ", prs.daq_dtypes)
    print("event length: ", prs.ev_len)
    prs.create_clean_file(**kwargs)


def parse_to_hdf5(binary_file_name, save_fname=None, n_anodes=16,
                  trace_fields=('cathode_trace', 'timing_trace'), **kwargs):
    """Parses a clean raw binary file to a hdf5 file."""

    prs = evt_parser88.Parser(n_anodes=n_anodes)
    prs.load_binary_file(binary_file_name)
    samples = prs.raw_samples
    prs.daq_dtypes = create_new_daq_dtype(samples, **kwargs)
    daq_dtypes = prs.daq_dtypes  # ensures it is set and they are the same
    print("Daq_dtypes in parse to hdf5: ", prs.daq_dtypes)
    print("ev length: ", prs.ev_len)
    # trace_fields = ('cathode_trace', 'timing_trace')  # raw trace fields

    evt_names = [name for name in daq_dtypes.names if name not in trace_fields]
    evt_formats = [daq_dtypes[daq_dtypes.names.index(name)] for name in daq_dtypes.names if name not in trace_fields]
    evt_dtypes = np.dtype({"names": evt_names, "formats": evt_formats})

    trace_names = [name for name in daq_dtypes.names if name in trace_fields]
    trace_formats = [daq_dtypes[daq_dtypes.names.index(name)] for name in daq_dtypes.names if name in trace_fields]

    if save_fname is None:
        save_fname = os.path.join(os.getcwd(), '../../Data_hdf5', os.path.splitext(os.path.basename(binary_file_name))[0] + '.h5')
    makedirs(save_fname)

    h5file = tables.open_file(save_fname, mode="w", title="Acquisition Data File")  # event fields

    h5file.create_table('/', 'event_data', description=evt_dtypes)
    for trace_name, trace_format in zip(trace_names, trace_formats):  # raw traces
        print("trace name: ", trace_name)
        print("trace format: ", trace_format)
        print("atom: ", tables.Atom.from_dtype(trace_format))
        # h5file.create_earray('/', trace_name, atom=tables.Atom.from_dtype(trace_format))  # TODO: What is wrong?
        h5file.create_earray('/', trace_name, atom=tables.UInt16Atom(), shape=(0, samples))

    h5file.flush()
    base_node = '/'  # The "home" directory of the file

    for ret in prs.batch_parse(return_dict=False):  # Just get back raw arrays
        evts = ret.shape[0]

        for table in h5file.iter_nodes(base_node, classname='Table'):  # Structured Data_hdf5 sets
            # print("Table description:", table.description._v_dtype)
            data_struct = np.zeros(evts, dtype=table.description._v_dtype)
            for field in table.description._v_names:  # Field functions as a key
                if ret[field] is not None:
                    # print("Field:", field)
                    # print("Data Dict[field]:", data_dict[field])
                    data_struct[field] = ret[field]
            table.append(data_struct)
            table.flush()

        for earray in h5file.iter_nodes(base_node, classname='EArray'):  # Homogeneous Data_hdf5 sets
            earray.append(ret[earray.name])
            earray.flush()

        h5file.flush()

    h5file.flush()
    h5file.close()
    print("Hdf5 file successfully saved!")


def preprocess_main():  # binary to (clean) binary
    # base_folder = "C:/Users/tlbr-user/Documents/TlBr_daq_analysis_wc/LocalDigitizerData/"
    base_folder = "C:/Users/justi/Documents/GitHub/TlBr/sample_data/LocalDigitizerData/"  # personal windows
    # data_file_name = "DavisD2022_9_22T16_3.dat"  # smaller Data_hdf5 fields
    # data_file_name = "DavisD2022_9_23T15_51.dat"  # weekend run, 1.7M events. Anode trigger fields but not cathode
    # data_file_name = "DavisD2022_9_27T13_52.dat"  # CG 8, FG 64, no sipm max
    # data_file_name = "DavisD2022_9_28T13_48.dat"  # CG 8, FG 0, cs-137, no sipm max, anode trigger in
    # data_file_name = "DavisD2022_9_28T15_7.dat"  # same as 9_28T15_7 but CG to 32 to see Cs137 features
    # data_file_name = "DavisD2022_9_28T16_13.dat"  # Th-228, CG 8, FG 0, no sipm max
    # data_file_name = "DavisD2022_9_30T13_54.dat"  # Co60, CG 8, FG 0, no sipm max

    # IEEE
    # data_file_name = "DavisD2022_10_20T16_3.dat"  # Na22, no SIPM?
    # data_file_name = "DavisD2022_10_21T11_11.dat"  # not sure 1
    # data_file_name = "DavisD2022_10_21T14_51.dat"  # not sure 2
    data_file_name = "DavisD2022_10_24T9_9.dat"  # Th228 SIPM

    fname = base_folder + data_file_name
    n_anodes = 16
    preprocess_binary_file(fname, n_anodes=n_anodes, verbose=True)


def main():  # clean binary to hdf5
    # base_folder = "C:/Users/tlbr-user/Documents/TlBr_daq_analysis_wc/LocalDigitizerData/"  # daq save folder
    # base_folder = "C:/Users/tlbr-user/Documents/TlBr_Analysis_Python/drs4timing/Data_processed_binary/"
    base_folder = "C:/Users/justi/Documents/GitHub/TlBr/code/depth_correction/Data_processed_binary/"  # Na22, no SIPM?
    # data_file_name = "DavisD2022_9_20T17_1_clean.dat"  # ~200k events, overnight, coarse gain = 16, fine gain 127
    # data_file_name = "DavisD2022_9_22T16_3_clean.dat"
    # data_file_name = "DavisD2022_9_22T16_3.dat"
    # data_file_name = "DavisD2022_9_23T15_51_clean.dat" # weekend run, 1.7M events. Anode trigger fields but no cathode
    # data_file_name = "DavisD2022_9_27T13_52_clean.dat"  # CG 8, FG 64, no sipm max
    # data_file_name = "DavisD2022_9_28T13_48_clean.dat"  # CG 8, FG 0, cs-137, no sipm max, anode trigger in
    # data_file_name = "DavisD2022_9_28T15_7_clean.dat" # same as 9_28T15_7 but CG to 32 to see Cs137 features
    # data_file_name = "DavisD2022_9_28T16_13_clean.dat"  # Th-228, CG 8, FG 0, no sipm max
    # data_file_name = "DavisD2022_9_30T13_54_clean.dat"  # Co60, CG 8, FG 0, no sipm max
    # data_file_name = "DavisD2022_10_20T16_3_clean.dat"  # Probably larger Cs137 data
    # data_file_name = "DavisD2022_10_21T11_11_clean.dat"  # not sure 1
    # data_file_name = "DavisD2022_10_21T14_51_clean.dat"  # not sure 2
    data_file_name = "DavisD2022_10_24T9_9_clean.dat"  # Th228 SIPM

    fname = base_folder + data_file_name
    n_anodes = 16
    parse_to_hdf5(fname, n_anodes=n_anodes)


def process_to_hdf5(raw_binary_file_name):  # raw binary to clean binary to hdf5
    import os
    # from pathlib import Path

    # raw_binary_base_folder = "C:/Users/tlbr-user/Documents/TlBr_Analysis_Python/drs4timing/Data_processed_binary/"
    # raw_binary_base_folder = "C:/Users/justi/Documents/GitHub/TlBr/code/depth_correction/Data_processed_binary/"  # Na22, no SIPM?
    raw_binary_base_folder = "C:/Users/justi/Documents/GitHub/TlBr/sample_data/LocalDigitizerDataCrocker/"
    # Windows personal PC
    # raw_binary_file_name = "DavisD2022_9_30T13_54.dat"  # Co60, CG 8, FG 0, no sipm max
    clean_binary_file_name = os.path.join(os.getcwd(), "Data_processed_binary",
                                      os.path.splitext(os.path.basename(raw_binary_file_name))[0] + '_clean.dat')
    # TODO: Return the file names directly from methods
    n_anodes = 16
    preprocess_binary_file(raw_binary_base_folder + raw_binary_file_name, n_anodes=n_anodes, verbose=True)
    print("Raw binary to clean binary complete. Converting to hdf5!")
    parse_to_hdf5(clean_binary_file_name, n_anodes=n_anodes)


if __name__ == "__main__":
    # stage position data at Crocker
    # raw_bin_fname = "DavisD2022_10_17T13_57.dat"
    # raw_bin_fname = "DavisD2022_10_17T14_10.dat"
    # raw_bin_fname = "DavisD2022_10_17T14_25.dat"
    # raw_bin_fname = "DavisD2022_10_17T14_39.dat"
    # raw_bin_fname = "DavisD2022_10_17T14_52.dat"
    # raw_bin_fname = "DavisD2022_10_17T15_6.dat"

    # overnight data Crocker
    # raw_bin_fname = "DavisD2022_10_16T18_26.dat"

    # November 1-3 Na-22 Cherenkov Data Run at Davis
    # raw_bin_fname = "DavisD2022_11_1T11_37.dat"

    # Last Na-22 run SIPM/Charge Readout at Davis, high threshold trigger (0.4 V)
    raw_bin_fname = "DavisD2022_11_3T14_52.dat"

    # preprocess_main()
    # main()
    process_to_hdf5(raw_bin_fname)
