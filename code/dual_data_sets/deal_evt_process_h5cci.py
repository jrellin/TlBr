import numpy as np
import os
from dds_utils import load_h5file, makedirs
import tables
from binio import DRS4BinaryFile
# Same as dual_evt_process but CCI file is already in h5


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
        # self.n_anodes = n_anodes
        # self.det_calibration = np.ones(self.n_anodes)  # set this otherwise, set to physical location
        # self.n_cathodes = n_cathodes
        self.ev_len = None
        self._cci_dtypes = None
        self.sipm_dtypes = None
        self.cci_h5file_loaded = False
        self.sipm_binary_loaded = False
        self.verbose = verbose
        self.cci_fname = None
        self.cci_file = None
        self.cci_evt_table = None
        self.sipm_file = None
        self.data_collection_type = None

    @property
    def cci_dtypes(self):
        return self._cci_dtypes

    @cci_dtypes.setter
    def cci_dtypes(self, val):
        self._cci_dtypes = val

    def load_cci_h5_file(self, fname):
        self.cci_fname = fname
        self.cci_file = load_h5file(fname)  # h5 file obj
        print("Print cci file:")
        print(self.cci_file)
        self.cci_h5file_loaded = True
        self.cci_evt_table = self.cci_file.root.event_data
        self.cci_dtypes = self.cci_evt_table.description._v_dtype
        print("CCI data types: ", self.cci_dtypes)
        print("nrows cci event data: ", self.cci_evt_table.nrows)

    def load_drs4_binary_file(self, drs4_fname, ch=4):
        if ch not in (1, 4):
            ValueError('{ch} must be either 1 (button source measurmeents) or 4 (Crocker).'.format(ch=ch))
        if ch == 4:  # 4 channel
            print("4 Channel")
            from sipm_4ch_drs4 import SiPM4
            from dds_utils import sipm_4c_data_types

            self.sipm_dtypes = sipm_4c_data_types()
            self.sipm_file = SiPM4(drs4_fname)
            self.data_collection_type = "Crocker"
        else:  # 1 channel
            from sipm_1ch_drs4 import SiPM1
            from dds_utils import sipm_1c_data_types

            self.sipm_dtypes = sipm_1c_data_types()
            self.sipm_file = SiPM1(drs4_fname)
            self.data_collection_type = "GBSF"
        self.sipm_binary_loaded = True

    def correlate_data_sets(self, output_fname=None):
        if not (self.cci_h5file_loaded & self.sipm_binary_loaded):
            raise RuntimeError("CCI h5 or SIPM (DRS4) binary file not loaded!")
        if output_fname is None:
            output_fname = os.path.join(os.getcwd(), 'Data_correlated',
                                        os.path.splitext(os.path.basename(self.cci_fname))[0] + '_' +
                                        self.data_collection_type + '_combined.h5')
        makedirs(output_fname)

        h5file = tables.open_file(output_fname, mode="w", title="Correlated Data File")  # event fields
        cci_cor_table = h5file.create_table('/', 'cci_event_data', description=self.cci_dtypes)
        sipm_cor_table = h5file.create_table('/', 'sipm_event_data', description=self.sipm_dtypes)
        h5file.flush()

        cci_cur_ptr = 0
        sipm_evt = self.sipm_file.process_sipm_evt(d_f=0.2, increment_to_next_event=True)
        # TODO: Allow changing of t0_f

        sipm_cur_id = sipm_evt['drs4_evt_id']

        keep_reading = True

        try:
            while keep_reading & (cci_cur_ptr < self.cci_evt_table.nrows):
                cci_evt = self.cci_evt_table.read(cci_cur_ptr, cci_cur_ptr + 1)
                cci_cur_id = cci_evt['event_counter'][0]

                if cci_cur_id < sipm_cur_id:  # Need to just increment CCI data
                    cci_cur_ptr += 1
                    continue

                if cci_cur_id > sipm_cur_id:
                    # increment sipm, backtrack cci to worse case (difference between the cci and sipm counters)
                    backtrack = int(cci_cur_id - sipm_cur_id)
                    sipm_evt = self.sipm_file.process_sipm_evt(d_f=0.2, increment_to_next_event=True)
                    sipm_cur_id = sipm_evt['drs4_evt_id']
                    if (cci_cur_ptr - backtrack) < 0:
                        cci_cur_ptr = 0
                    else:
                        cci_cur_ptr -= backtrack
                    continue

                if cci_cur_id == sipm_cur_id:  # save both
                    cci_cor_table.append(cci_evt)
                    cci_cor_table.flush()

                    sipm_cor_evt = np.zeros(1, dtype=self.sipm_dtypes)
                    # sipm_cor_evt['rf_zero_cs'] = 0
                    # print("sipm cor evt zeros: ", sipm_cor_evt)
                    # print("zeros dtypes: ", sipm_cor_evt.dtype)

                    for field in sipm_cor_table.description._v_names:  # Field functions as a key
                        if sipm_evt[field] is not None:
                            # print("Field: ", field)
                            # print("sipm_cor_evt[field]: ", sipm_cor_evt[field])
                            # print("sipm_evt[field]: ", sipm_evt[field])
                            sipm_cor_evt[field] = sipm_evt[field]
                    sipm_cor_table.append(sipm_cor_evt)
                    sipm_cor_table.flush()

                    h5file.flush()
                    cci_cur_ptr += 1
                    sipm_evt = self.sipm_file.process_sipm_evt(d_f=0.2, increment_to_next_event=True)
                    sipm_cur_id = sipm_evt['drs4_evt_id']

                # sipm_evt = self.sipm_file.process_sipm_evt(df=0.2, increment_to_next_event=True)
        except StopIteration:
            print("Reached last event in SIPM file!")
            keep_reading = False
            h5file.flush()
            h5file.close()

        print("Reached end of correlating!")
        # h5file.flush()
        # h5file.close()
        self.cci_file.close()


def main_crocker():
    import os
    from pathlib import Path

    # cci_h5_data_file = "DavisD2022_10_17T13_57_clean.h5"  # p0
    # cci_h5_data_file = "DavisD2022_10_17T14_10_clean.h5"  # p2
    # cci_h5_data_file = "DavisD2022_10_17T14_25_clean.h5"  # p4
    # cci_h5_data_file = "DavisD2022_10_17T15_6_clean.h5"  # p5, this one was done LAST, look at time (15_6)
    cci_h5_data_file = "DavisD2022_10_17T14_39_clean.h5"  # p6
    # cci_h5_data_file = "DavisD2022_10_17T14_52_clean.h5"  # p8
    cci_h5_full_fname = os.path.join(str(Path(os.getcwd()).parents[1]), "Data_hdf5",  cci_h5_data_file)

    # sipm_binary_data_file = "20221017_Crocker_31.6V_cherenkov_500pa_DualDataset_nim_amp_p0_v9.dat"  # p0
    # sipm_binary_data_file = "20221017_Crocker_31.6V_cherenkov_500pa_DualDataset_nim_amp_p2_v10.dat"  # p2
    # sipm_binary_data_file = "20221017_Crocker_31.6V_cherenkov_500pa_DualDataset_nim_amp_p4_v11.dat"  # p4
    # sipm_binary_data_file = "20221017_Crocker_31.6V_cherenkov_500pa_DualDataset_nim_amp_p5_v14.dat"  # p5
    sipm_binary_data_file = "20221017_Crocker_31.6V_cherenkov_500pa_DualDataset_nim_amp_p6_v12.dat"  # p6
    # sipm_binary_data_file = "20221017_Crocker_31.6V_cherenkov_500pa_DualDataset_nim_amp_p8_v13.dat"  # p8
    sipm_full_fname = os.path.join(str(Path(os.getcwd()).parents[1]), "sample_data", "drs4", sipm_binary_data_file)

    ch = 4
    cor_data = Combiner()
    cor_data.load_drs4_binary_file(sipm_full_fname, ch=ch)
    cor_data.load_cci_h5_file(cci_h5_full_fname)
    cor_data.correlate_data_sets()


def main_GBSF():  # button source data
    import os
    from pathlib import Path

    # cci_h5_data_file = "DavisD2022_11_1T11_37_clean.h5"  # Na-22 Data Set 1, det 2, SiPM
    cci_h5_data_file = "DavisD2022_11_3T14_52_clean.h5" # NA-22, Set 3 MUCH higher threshold
    cci_h5_full_fname = os.path.join(str(Path(os.getcwd()).parents[1]), "Data_hdf5", cci_h5_data_file)

    # sipm_binary_data_file = "20221031_Davis_40_5V_Thr20mV_IEEE_Na22_DualDataset.dat"  # Na-22 Data Set 1, det 2, SiPM
    sipm_binary_data_file = "20221103_Davis_40_5V_Thr20mV_IEEE_Na22_DualDataset3.dat"  # NA-22, Set 3 MUCH higher threshold
    sipm_full_fname = os.path.join(str(Path(os.getcwd()).parents[1]), "sample_data", "drs4", sipm_binary_data_file)

    ch = 1
    cor_data = Combiner()
    cor_data.load_drs4_binary_file(sipm_full_fname, ch=ch)
    cor_data.load_cci_h5_file(cci_h5_full_fname)
    cor_data.correlate_data_sets()


if __name__ == "__main__":
    # main_crocker()
    main_GBSF()

