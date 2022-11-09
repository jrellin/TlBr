import numpy as np
import tables


def linear_interpolate_trigger(time_bins, waveform, baseline, f=np.array([0.2]), ret_max_instead=False):
    """Assumes positive polarity signals. Trying to fix spike issue seen in lf2 p2 1200 skip trigger. Used in crocker_energy_timing_IEEE.py"""
    wf = waveform - baseline
    t = time_bins
    max_sig_fv = f * np.max(wf)

    ati = np.argmax(wf) - np.argmin(wf[:np.argmax(wf)][::-1] >= max_sig_fv)
    m = (wf[ati] - wf[ati-1]) / (t[ati]-t[ati-1])  # slope
    if m != 0:
        interp_t = t[ati-1] + ((max_sig_fv - wf[ati-1])/m)
    else:
        interp_t = t[ati - 1]  # nothing to interp

    if ret_max_instead:
        return interp_t, np.max(wf) + baseline  # add back baseline for plotting
    return interp_t, max_sig_fv + baseline  # add back baseline for plotting


def sipm_4c_data_types():
    # ds16 = np.dtype(np.int16).newbyteorder('<')  # signed 16 integer
    # d16 = np.dtype(np.uint16).newbyteorder('<')
    d32 = np.dtype(np.uint32).newbyteorder('<')
    # d64 = np.dtype(np.uint64).newbyteorder('<')
    d32f = np.dtype(np.single).newbyteorder('<')
    # d64f = np.dtype(np.double).newbyteorder('<')

    data_fields = ['drs4_evt_id',
                   'rf_zero_cs', 'rf_zero_cs_signs',
                   't0_trigs', 't0_v_at_trig',
                   'det_trig', 'det_v_at_trig',
                   'det_en_peak', 'det_en_integral']

    data_types = [d32,
                  (d32f, (12,)), (d32f, (12,)),
                  (d32f, (5,)), (d32f, (5,)),  # -1 in empty spaces
                  d32f, d32f,
                  d32f, d32f]

    daq_dtypes = np.dtype({"names": data_fields, "formats": data_types})
    return daq_dtypes


def sipm_1c_data_types():
    d32 = np.dtype(np.uint32).newbyteorder('<')  # 32 bit unsigned int
    d32f = np.dtype(np.single).newbyteorder('<')  # 32 bit float

    data_fields = ['drs4_evt_id', 'det_trig', 'det_v_at_trig', 'det_en_peak', 'det_en_integral']
    data_types = [d32, d32f, d32f, d32f, d32f]
    daq_dtypes = np.dtype({"names": data_fields, "formats": data_types})
    return daq_dtypes


def makedirs(path):
    """ Create directories for `path` (like 'mkdir -p'). """
    import os
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
