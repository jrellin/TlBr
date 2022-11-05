import numpy as np
from analysis_tools import load_h5file
import matplotlib.pyplot as plt
# Crocker charge induction signal, not integration yet with cherenkov


class Analysis(object):
    """This class performs on the fly parsing of Data_hdf5 from SIS3316 card(s) based on set config settings"""
    n_anodes = 16
    n_cathodes = 1
    block_read_evts = 100000
    sample_period = 16/1000  # digitizer sample rate in microseconds

    def __init__(self, filename, verbose=False):
        self.det_calibration = np.ones(self.n_anodes)  # set this otherwise, set to physical location
        self.f = load_h5file(filename)
        self.evt_data = self.f.root.event_data
        self.samples = self.evt_data.col('points')[0]
        self.evts = self.evt_data.nrows
        self.filtered_evts = 0

        self.pin_map = np.array([[2, 1, 3, 4], [15, 8, 6, 5], [13, 11, 9, 7], [16, 14, 12, 10]])
        # channel map as pins as plugged into digitizer
        self.det_map = np.arange(1, 16+1).reshape([4, 4])
        # channel map as located in space, "top left" is channel 1, 16 is "bottom right" with TlBr
        # in upper left corner of pixel board
        self.max_energy_time_histogram = None  # Eventually ETImage Object
        self.anodes_energy_time_histograms = None  # Eventually list of ETImage Objects, 1 for each anode
        self.verbose = verbose

    def __del__(self):
        # close file
        self.f.close()

    def close(self):
        self.f.close()

    def _filter_overflow_anode_events(self, curr_idx, end_idx):
        filter_str = '(amp1 < 4095)'  # start
        for n in np.arange(2, self.n_anodes + 1):
            filter_str += '& (amp' + str(n) + '< 4095) '
        filter_str = filter_str[:-1]  # remove that last space

        anode_trgs = np.array([row['anode_trgs']
                               for row in self.evt_data.where(filter_str, start=curr_idx, stop=end_idx)])

        post_flt_evts = anode_trgs.shape[0]
        self.filtered_evts += (end_idx - curr_idx) - post_flt_evts  # how many events were REMOVED
        corrected = np.zeros([self.n_anodes, post_flt_evts])

        for n in np.arange(1, self.n_anodes + 1):
            det_ch = self.det_map[self.pin_map == n].item()  # maps digitizer channel to physical location
            corrected[det_ch - 1, :] = np.array(
                [self.det_calibration[det_ch - 1] * (row['amp' + str(n)] - row['bl' + str(n)])
                 for row in self.evt_data.where(filter_str, start=curr_idx, stop=end_idx)])

        return corrected, anode_trgs

    def _no_filter_overflow_anode_events(self, curr_idx, end_idx, corrected):  # Need corrected array
        evts_idx = np.arange(curr_idx, end_idx)  # allows filtering of events
        for n in np.arange(1, self.n_anodes + 1):
            det_ch = self.det_map[self.pin_map == n].item()  # maps digitizer channel to physical location
            det_amp = self.evt_data.col('amp' + str(n))[evts_idx]
            det_bl = self.evt_data.col('bl' + str(n))[evts_idx]
            corrected[det_ch - 1, :] = self.det_calibration[det_ch - 1] * (det_amp - det_bl).astype('float')
            # corrected[det_ch - 1, :] = self.det_calibration[det_ch - 1] * det_amp.astype('float') # no bl sub.

        anode_trgs = self.evt_data.col('anode_trgs')[evts_idx]
        return corrected, anode_trgs

    def create_timing_2d_histograms(self, energy_limits=None, time_limits=None,
                                     energy_bins=(2**12)//8, time_bins=1000//2, filter_overflow=True,
                                    time_cut=False, time_cut_limits=(5, 20)):
        """Create a 2D histogram from the anode and timing signals. Cherenkov triggering. Energy limits are by bin,
        time limits are in microseconds. """
        time_bounds = np.array([0, 1.0])  # TODO: Maybe detect from limits if its meant as a fraction?
        energy_bounds = np.array([0, 1.0])

        if time_limits is not None:  # assume in microseconds
            time_bounds = (time_limits / (self.sample_period * self.samples))  # fractions of total samples in trace

        if energy_limits is not None:
            energy_bounds = energy_limits / (2**12)  # fractions of ADC range

        time_range = (time_bounds * self.samples).astype('int')
        energy_range = (energy_bounds * (2 ** 12)).astype('int')

        max_img = ETImage([energy_range, time_range], [energy_bins, time_bins])
        anode_imgs = [ETImage([energy_range, time_range], [energy_bins, time_bins]) for _ in np.arange(self.n_anodes)]

        curr_idx = 0
        corrected = np.zeros([self.n_anodes, self.block_read_evts])  # temporary array
        self.filtered_evts = 0

        while curr_idx < self.evts:
            if curr_idx + self.block_read_evts > self.evts:
                end_idx = self.evts
                corrected = np.zeros([self.n_anodes, self.evts - curr_idx])
            else:
                end_idx = curr_idx + self.block_read_evts

            if filter_overflow:
                corrected, anode_trgs = self._filter_overflow_anode_events(curr_idx, end_idx)
            else:
                corrected, anode_trgs = self._no_filter_overflow_anode_events(curr_idx, end_idx, corrected)

            max_anode_vals = np.max(corrected, axis=0)
            # last_timing = self.evt_data.col('last_timing')[curr_idx:end_idx]

            if not time_cut:
                # BEFORE START, last trigger
                # last_timing = np.max(anode_trgs, axis=1)
                # last_timing[last_timing < 0] = 0
                # BEFORE END

                # first trigger
                anode_trgs[anode_trgs < 0] = 12000
                timing = np.min(anode_trgs, axis=1)
                # print("Time range [1]: ", time_range[1])
                timing[timing > time_range[1]] = time_range[1]
                filt_trgs = np.ones(timing.size, dtype=bool)
                max_img.add_values_to_image(max_anode_vals, timing)
            else:
                anode_trgs[anode_trgs < 0] = 12000  # first trigger
                timing = np.min(anode_trgs, axis=1)
                filt_trgs = (timing < time_range[1]) & (timing > time_cut_limits[0]/self.sample_period) & (timing < time_cut_limits[1]/self.sample_period)
                max_img.add_values_to_image(max_anode_vals[filt_trgs], timing[filt_trgs])

            # last_timing = np.min(anode_trgs, axis=1)
            # max_img.add_values_to_image(max_anode_vals, last_timing)
            t_filt_corrected = corrected[:, filt_trgs]

            for n in np.arange(1, self.n_anodes + 1):  # n is channel id
                idx = n - 1  # python indexes at 0
                # ch_max_idx = np.argwhere(np.argmax(corrected, axis=0) == idx)  # i.e. where ch is max
                timing_filtered = timing[filt_trgs]

                ch_max_idx = np.argwhere(np.argmax(t_filt_corrected, axis=0) == idx)  # i.e. where ch is max
                ch_max_vals = corrected[idx, ch_max_idx].flatten()
                ch_max_timing = timing_filtered[ch_max_idx].flatten()
                anode_imgs[idx].add_values_to_image(ch_max_vals, ch_max_timing)


            print("{x} events read!".format(x=curr_idx + self.block_read_evts))
            curr_idx += self.block_read_evts

        print("All events read!")
        print("Max hist sum: ", max_img.img.sum())

        self.max_energy_time_histogram, self.anodes_energy_time_histograms = max_img, anode_imgs

    def plot_2d_timing_histograms(self, pretrigger=0.05, plot_trigger=True, **kwargs):
        # self._create_timing_2d_histograms(**kwargs)
        max_img, anode_imgs = self.max_energy_time_histogram, self.anodes_energy_time_histograms
        # energy_limits=None, time_limits=None, time_bins=1000
        en_edges, t_edges = max_img.bins

        x_extent = (en_edges[0], en_edges[-1])
        y_extent = (t_edges[0] * self.sample_period, t_edges[-1] * self.sample_period)
        # y_extent = (t_edges[0], t_edges[-1])

        print("x extent: ", x_extent)
        print("y extent: ", y_extent)

        nrow = self.n_anodes // 4
        ncol = self.n_anodes // 4

        fig1, axs1 = plt.subplots(nrow, ncol, figsize=(16, 12))
        for i, (ax, anode_hist) in enumerate(zip(fig1.axes, anode_imgs)):
            ch_img = anode_hist.img
            ch_img[0, :] = 0
            ch_img[-1,: ] = 0
            # ch_img = np.log(ch_img, where=(ch_img > 0))  # TODO: Remove
            img = ax.imshow(ch_img, cmap='magma_r', origin='lower', interpolation='none', extent=np.append(x_extent, y_extent), aspect='auto')
            fig1.colorbar(img, fraction=0.046, pad=0.04, ax=ax)
            ax.set_xlabel('ch ' + str(i + 1) + ' ADC bin')
            ax.set_ylabel('timing')
            if plot_trigger:
                ax.axhline(y=pretrigger * self.samples * self.sample_period, color='g', linestyle='--')
            ax.set_xlim((0, 1200))
            ax.set_ylim(46, 80)

        fig2, ax2 = plt.subplots(1, 1, figsize=(6, 6))
        m_img = max_img.img
        print("Max Hist filtered (percent):", m_img[0, :].sum() / m_img.sum())
        m_img[0, :] = 0
        m_img[-1, :] = 0
        img = ax2.imshow(m_img, cmap='magma_r', origin='lower', interpolation='none', extent=np.append(x_extent, y_extent), aspect='auto')
        fig2.colorbar(img, fraction=0.046, pad=0.04, ax=ax2)
        if plot_trigger:
            ax2.axhline(y=pretrigger * self.samples * self.sample_period, color='g', linestyle='--')
        # ax2.set_xlim((0, 600))
        # ax2.set_ylim(46, 50)
        ax2.set_xlabel('Max Anode ADC bin')
        ax2.set_ylabel('timing')

        fig1.tight_layout()
        fig2.tight_layout()

        plt.show()

    def plot_1D_projections(self, pretrigger=0.05, plot_trigger=True,  energy_log_scale=True, time_log_scale=False,
                            energy_calib=True):
        # self._create_timing_2d_histograms(**kwargs)
        max_hist, anode_histograms = self.max_energy_time_histogram, self.anodes_energy_time_histograms
        en_edges, t_edges = max_hist.bins

        print("Max hist shape: ", max_hist.img.shape)
        print("max hist sum ax 0: ", np.sum(max_hist.img, axis=0).shape)
        en_step_bins = (en_edges[1:] + en_edges[:-1])/2
        t_step_bins = (t_edges[1:] + t_edges[:-1])/2

        print("en_steps: ", en_step_bins.size)
        print("t steps: ", t_step_bins.size)

        nrow = self.n_anodes // 4
        ncol = self.n_anodes // 4

        fig1, axs1 = plt.subplots(nrow, ncol, figsize=(16, 12))
        fig2, axs2 = plt.subplots(nrow, ncol, figsize=(16, 12))

        for i, (e_ax, t_ax, ch_obj) in enumerate(zip(fig1.axes, fig2.axes, anode_histograms)):  # energy

            ch_hist = ch_obj.img
            # ch_hist[0, :] = 0

            en_proj = np.sum(ch_hist, axis=0)  # energy projection
            t_proj = np.sum(ch_hist, axis=1)  # time projection

            if energy_calib:
                b_to_en = 511 / 670 # * energy_scaling
                e_ax.step(en_step_bins * b_to_en, en_proj, 'b-', where='mid')
                e_ax.set_xlabel('Energy (keV))')
                # a_ax.set_xlim((0, 800))  # Cs137, shifted
                # e_ax.set_xlim((0, 1600))  # Na22, shifted
            else:
                e_ax.step(en_step_bins, en_proj, 'b-', where='mid')
                e_ax.set_xlabel('Anode ADC')

            # e_ax.step(en_step_bins, en_proj, 'b-', where='mid')
            t_ax.step(t_step_bins * self.sample_period, t_proj, 'b-', where='mid')

            e_ax.set_title("Amplitude (Channel {n})".format(n=i + 1))
            t_ax.set_title("Amplitude (Channel {n})".format(n=i + 1))
            # ax.set_xlabel('ch ' + str(i+1))
            e_ax.set_ylabel('Counts')
            t_ax.set_ylabel('Counts')

            if energy_log_scale:
                e_ax.set_yscale('log')

            if time_log_scale:
                t_ax.set_yscale('log')

            if plot_trigger:
                t_ax.axvline(x=pretrigger * self.samples * self.sample_period, color='g', linestyle='--')

        fig1.tight_layout()
        fig2.tight_layout()

        fig3, (ax31, ax32) = plt.subplots(1, 2, figsize=(16, 12))  # ax31 -> energy, ax32 -> time
        max_img = max_hist.img
        # max_img[0, :] = 0
        e_max_proj = np.sum(max_img, axis=0)  # energy max ch projection
        t_max_proj = np.sum(max_img, axis=1)  # time max ch projection

        if energy_calib:
            b_to_en = 511 / 670 # * energy_scaling
            ax31.step(en_step_bins * b_to_en, e_max_proj, 'b-', where='mid')
            ax31.set_xlabel('Anode (keV))')
            # ax31.set_xlim((0, 800))  # Cs137, shifted
            # ax31.set_xlim((0, 1600))  # Na22, shifted
        else:
            ax31.step(en_step_bins, e_max_proj, 'b-', where='mid')
            ax31.set_xlabel('Anode ADC')

        # ax31.step(en_step_bins, e_max_proj, 'b-', where='mid')
        ax32.step(t_step_bins * self.sample_period, t_max_proj, 'b-', where='mid')

        ax31.set_ylabel('Counts')
        ax32.set_ylabel('Counts')

        ax31.set_xlabel('Energy Bin')
        ax32.set_xlabel('Time (us))')

        if energy_log_scale:
            ax31.set_yscale('log')  # energy

        if time_log_scale:
            ax32.set_yscale('log')  # time

        # ax32.set_xlim([35, 100])
        if plot_trigger:
            ax32.axvline(x=pretrigger * self.samples * self.sample_period, color='g', linestyle='--')

        fig3.tight_layout()

        plt.show()

    def plot_energy_by_timing_slices(self, time_start, delta_t, n_bins):
        """Overlays multiple time slices of generated 2d time energy histogram. Time start, delta_t are in
        microseconds. Checks if n bins from time_start is in range, throws error otherwise."""
        max_img, (e_bins, t_bins) = self.max_energy_time_histogram.img, self.max_energy_time_histogram.bins
        # n_e_bins = e_bins.size - 1  # e_bins is really e_bin_edges
        print("max_img dimensions: ", max_img.shape)

        en_step_bins = (e_bins[1:] + e_bins[:-1]) / 2  # used for plotting
        gen2d_t_width = (t_bins[1] - t_bins[0])
        # This is the time width of each bin in generated 2d histogram in number of samples
        # TODO: Allow for non-uniform binning

        t_start_bin = time_start/self.sample_period
        t_bin_width = np.max((delta_t/self.sample_period, 1))  # bin width can't be less than sample period
        t_end_bin = (t_start_bin + (t_bin_width * n_bins))

        if t_start_bin < t_bins[0]:
            raise ValueError("Time start {ts} microseconds is less than min extent of {hs} microseconds "
                             "in generated 2D histogram.".format(ts=time_start, hs=t_bins[0] * self.sample_period))

        if t_end_bin > t_bins[-1]:
            raise ValueError("Time stop {ts} microseconds is greater than max extent of {hs} microseconds "
                             "in generated 2D histogram.".format(ts=time_start + (delta_t + n_bins),
                                                                 hs=t_bins[-1] * self.sample_period))

        t_start_scaled = int(t_start_bin/gen2d_t_width)
        t_bin_scaled_width = int(t_bin_width/gen2d_t_width)
        t_end_scaled = int(t_start_scaled + (n_bins * t_bin_scaled_width))

        print("Old bin width (microseconds): {ob}".format(ob=gen2d_t_width * self.sample_period))
        print("New bins")
        print("Starting bin: {st}. In microseconds: {t}".format(st=t_start_scaled, t=time_start))
        print("Width of bins in t dim: {bt}. In microseconds: {t}".format(bt=t_bin_scaled_width, t=delta_t))
        # print("Ending bin: {et}. In microseconds: {t}".format(et=t_end_scaled, t=t_end_bin*self.sample_period))

        summed_img = max_img[t_start_scaled:t_end_scaled, :].reshape((n_bins, t_bin_scaled_width, max_img.shape[1])).sum(axis=1)
        start_bin = np.arange(time_start, (time_start + n_bins * delta_t), delta_t)
        # The image is "upside down" (imshow origin set to lower for display) so first row is lowest t value

        fig, ax = plt.subplots(1, 1, figsize=(16, 12))  # ax31 -> energy, ax32 -> time

        for t, t_bin_hist in zip(start_bin, summed_img):
            # ax.step(en_step_bins, t_bin_hist, where='mid',
            #         label=('{start}-{end}' + r'$\mu$' + 's').format(start=t, end=t + t_bin_width))
            ax.plot(en_step_bins, t_bin_hist,
                    label=('{start}-{end}' + r'$\mu$' + 's').format(start=t, end=t + delta_t))

        ax.set_yscale('log')
        ax.set_xlabel('ADC Bin')
        ax.set_ylabel('Counts')
        ax.legend(loc='best')
        ax.set_title("Cherenkov Time Slice Energy Histograms")

        plt.show()

    def plot_SIPM_amplitudes(self, batch_read=30000):
        """Histogram SIPM max amplitudes"""
        bins = (2 ** 12)  # 12 bit ADC
        bin_edges = np.arange(bins + 1)
        self.block_read_evts = batch_read  # TODO: Fix or reset at end

        curr_idx = 0
        sipm_histogram = np.histogram([], bins=bin_edges)[0]

        self.filtered_evts = 0

        while curr_idx < self.evts:
            if curr_idx + self.block_read_evts > self.evts:
                end_idx = self.evts
            else:
                end_idx = curr_idx + self.block_read_evts

            evts_idx = np.arange(curr_idx, end_idx)  # allows filtering of events
            sipm_histogram += np.histogram(self.evt_data.col('shaped_sipm_max')[evts_idx], bins=bin_edges)[0]

            print("{x} events read!".format(x=curr_idx + self.block_read_evts))
            curr_idx += self.block_read_evts

        fig, ax = plt.subplots(1, 1, figsize=(16, 12))
        ax.step(np.arange(bins), sipm_histogram, 'b-', where='mid')
        ax.set_xlabel('shaped_sipm_max')
        ax.set_ylabel('Counts')
        ax.set_yscale('log')
        plt.show()


class ETImage(object):
    def __init__(self, ranges, bins):
        """X axis is energy, y axis is time, range=[energy_range, time_range], bins=(energy_bins, time_bins))"""
        img_hist, self.xedges, self.yedges = np.histogram2d([], [], range=ranges, bins=bins)
        self._img_hist = img_hist.T  # Transpose needed

    def add_values_to_image(self, energy_vals, time_vals):
        img_hist = np.histogram2d(energy_vals, time_vals, bins=[self.xedges, self.yedges])[0]
        self._img_hist += img_hist.T

    def combine_data(self, et2d_histogram):
        self._img_hist += et2d_histogram

    @property
    def img(self):
        return self._img_hist.copy()

    @property
    def bins(self):
        return self.xedges, self.yedges  # energy, time


def main(filename, **kwargs):
    import os
    from pathlib import Path
    fname = os.path.join(str(Path(os.getcwd()).parents[1]), "Data_hdf5", filename)
    print("fname: ", fname)
    prc = Analysis(fname, **kwargs)
    # Ch 2 is arbitrary for det 1
    coarse_gain_correction = np.array([1, 0.66, 1, 1,
                                       1, 1, 1, 1,
                                       1, 1, 1, 1,
                                       2, 1, 1, 1])
    fine_gain_correction = 717 / np.array([717, 614, 611, 614,
                                           717, 680, 611, 635,
                                           667, 676, 640, 635,
                                           811, 678, 632, 640])  # 50% fall off in 511 line
    prc.det_calibration = coarse_gain_correction * fine_gain_correction
    # prc.plot_2d_timing_histograms()
    # time_cut=False, time_cut_limits=(5, 20)
    pretrigger = 0.15
    prc.create_timing_2d_histograms(time_cut=True, time_cut_limits=(35, 60))
    prc.plot_2d_timing_histograms(pretrigger=pretrigger)
    prc.plot_1D_projections(pretrigger=pretrigger)
    # prc.plot_energy_by_timing_slices(46, 8, 5)

    # prc.plot_1D_projections()
    # prc.plot_SIPM_amplitudes()
    prc.close()


def main_combine_data(**kwargs):
    import os
    from pathlib import Path

    data_files = ["DavisD2022_10_17T13_57_clean.h5", "DavisD2022_10_17T14_10_clean.h5",
                  "DavisD2022_10_17T14_25_clean.h5", "DavisD2022_10_17T14_39_clean.h5",
                  "DavisD2022_10_17T14_52_clean.h5", "DavisD2022_10_17T15_6_clean.h5"]

    files_to_include = np.arange(len(data_files))

    coarse_gain_correction = np.array([1, 0.66, 1, 1,
                                       1, 1, 1, 1,
                                       1, 1, 1, 1,
                                       2, 1, 1, 1])
    fine_gain_correction = 717 / np.array([717, 614, 611, 614,
                                           717, 680, 611, 635,
                                           667, 676, 640, 635,
                                           811, 678, 632, 640])

    scale_gain = 0.5
    t_cut = True

    accumulator = Analysis(os.path.join(str(Path(os.getcwd()).parents[1]), "Data_hdf5",
                                        data_files[files_to_include[0]]), **kwargs)
    accumulator.det_calibration = coarse_gain_correction * fine_gain_correction * scale_gain
    accumulator.create_timing_2d_histograms(time_cut=t_cut)
    # self.max_energy_time_histogram, self.anodes_energy_time_histograms
    # Initialized with first data file
    # accumulator.plot_1D_projections()
    print("Accumulator Counts: ", accumulator.max_energy_time_histogram.img.sum())

    for i in files_to_include[1:]:
        cur_data_file = data_files[i]
        fname = os.path.join(str(Path(os.getcwd()).parents[1]), "Data_hdf5", cur_data_file)
        single_run = Analysis(fname)
        single_run.det_calibration = coarse_gain_correction * fine_gain_correction
        single_run.create_timing_2d_histograms(time_cut=t_cut)

        accumulator.max_energy_time_histogram.combine_data(single_run.max_energy_time_histogram.img)

        for n in np.arange(accumulator.n_anodes):
            accumulator.anodes_energy_time_histograms[n].combine_data(single_run.anodes_energy_time_histograms[n].img)

        single_run.close()

        print("Accumulator Counts: ", accumulator.max_energy_time_histogram.img.sum())

    print("Final Accumulator Counts: ", accumulator.max_energy_time_histogram.img.sum())
    # accumulator.plot_1D_projections(energy_scaling=scale_gain)
    accumulator.plot_1D_projections()
    accumulator.plot_2d_timing_histograms()
    accumulator.close()


if __name__ == "__main__":
    # import os
    # from pathlib import Path

    # base_folder = "C:/Users/justi/PycharmProjects/TlBr/Data_hdf5/"  # Windows Personal PC
    # base_folder = "C:/Users/tlbr-user/Documents/TlBr_Analysis_Python/drs4timing/Data_hdf5/"  # GBSF Windows
    # base_folder = "C:/Users/tlbr-user/Documents/TlBr_Analysis_Python/drs4timing/Data_hdf5/"
    # data_file_name = "DavisD2022_9_20T17_1_clean.h5"
    # data_file_name = "DavisD2022_9_22T16_3_clean.h5"
    # data_file_name = "DavisD2022_9_23T15_51_clean.h5"  # weekend run
    # data_file_name = "DavisD2022_9_28T13_48_clean.h5"  # Cs-137
    # data_file_name = "DavisD2022_9_28T16_13_clean.h5"  # Th-228, CG 8, FG 0, no sipm max
    # data_file_name = "DavisD2022_9_30T13_54_clean.h5" # Co60, CG 8, FG 0, no sipm max

    # IEEE
    # data_file_name = "DavisD2022_10_24T9_9_clean.h5"  # Na 22

    # Crocker
    # file_name = "DavisD2022_10_17T13_57_clean.h5"
    # Crocker overnight
    # file_name = "DavisD2022_10_16T18_26_clean.h5"

    # Davis
    file_name = "DavisD2022_11_1T11_37_clean.h5"  # Na-22 November 1-3 Run

    # fname = os.path.join(str(Path(os.getcwd()).parents[1]), "Data_hdf5", data_file_name)
    # print("fname: ", fname)

    main(file_name)
    # main_combine_data()

