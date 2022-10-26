import numpy as np
from analysis_tools import load_h5file
import matplotlib.pyplot as plt
from code.utils import gaussian_fitting_functions as gft
# This version attempts to further refine selection of events with timing windows and storing the 2D histogram in an
# array for easy plotting in other methods. Builds off version 2 to be faster.
# Previously v41 before 10/05/22


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
        self.anode_bin_scaler=1

        self.pin_map = np.array([[2, 1, 3, 4], [15, 8, 6, 5], [13, 11, 9, 7], [16, 14, 12, 10]])
        # channel map as pins as plugged into digitizer
        self.det_map = np.arange(1, 16+1).reshape([4, 4])
        # channel map as located in space, "top left" is channel 1, 16 is "bottom right" with TlBr
        # in upper left corner of pixel board
        self.max_anode_cathode_histogram = None  # Eventually ETImage Object
        self.anode_cathode_histograms = None  # Eventually list of ETImage Objects, 1 for each anode
        self.verbose = verbose

    def __del__(self):
        # close file
        self.f.close()

    def close(self):
        self.f.close()

    def _filter_overflow_anode_events(self, curr_idx, end_idx):  # return cathode values and anod values
        filter_str = '(amp1 < 4095)'  # start
        for n in np.arange(2, self.n_anodes + 1):
            filter_str += '& (amp' + str(n) + '< 4095) '
        filter_str = filter_str[:-1]  # remove that last space

        cathode_amp = np.array([row['shaped_cathode_max']
                               for row in self.evt_data.where(filter_str, start=curr_idx, stop=end_idx)])

        post_flt_evts = cathode_amp.shape[0]
        self.filtered_evts += (end_idx - curr_idx) - post_flt_evts  # how many events were REMOVED
        corrected = np.zeros([self.n_anodes, post_flt_evts])

        for n in np.arange(1, self.n_anodes + 1):
            det_ch = self.det_map[self.pin_map == n].item()  # maps digitizer channel to physical location
            corrected[det_ch - 1, :] = np.array(
                [self.det_calibration[det_ch - 1] * (row['amp' + str(n)] - row['bl' + str(n)])
                 for row in self.evt_data.where(filter_str, start=curr_idx, stop=end_idx)])

        return corrected, cathode_amp

    def _no_filter_overflow_anode_events(self, curr_idx, end_idx, corrected):  # Need corrected array
        evts_idx = np.arange(curr_idx, end_idx)  # allows filtering of events
        for n in np.arange(1, self.n_anodes + 1):
            det_ch = self.det_map[self.pin_map == n].item()  # maps digitizer channel to physical location
            det_amp = self.evt_data.col('amp' + str(n))[evts_idx]
            det_bl = self.evt_data.col('bl' + str(n))[evts_idx]
            corrected[det_ch - 1, :] = self.det_calibration[det_ch - 1] * (det_amp - det_bl).astype('float')
            # corrected[det_ch - 1, :] = self.det_calibration[det_ch - 1] * det_amp.astype('float') # no bl sub.

        cathode_amp = self.evt_data.col('anode_trgs')[evts_idx]
        return corrected, cathode_amp

    def create_ca_2d_histograms(self, anode_range=None, cathode_range=None,
                                    anode_bins=None, cathode_bins=None, filter_overflow=True):
        """Create a 2D histogram from the anode and cathode signals. Range in number of ADC bins. """

        if anode_bins is None:
            anode_bins = (2**12)//4
            # anode_bins = (2 ** 12)

        self.anode_bin_scaler = (2**12)/anode_bins

        if cathode_bins is None:
            cathode_bins = (2**12)//4

        if anode_range is None:
            anode_range = (np.array([0, 1]) * 2**12).astype('int')

        if cathode_range is None:
            cathode_range = (np.array([0, 1]) * 2**12).astype('int')

        max_car_img = CAImage([anode_range, cathode_range], [anode_bins, cathode_bins])
        anode_car_imgs = [CAImage([anode_range, cathode_range], [anode_bins, cathode_bins]) for _ in np.arange(self.n_anodes)]

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
                corrected, cathode_amp = self._filter_overflow_anode_events(curr_idx, end_idx)
            else:
                corrected, cathode_amp = self._no_filter_overflow_anode_events(curr_idx, end_idx, corrected)

            max_anode_vals = np.max(corrected, axis=0)
            # last_timing = self.evt_data.col('last_timing')[curr_idx:end_idx]
            # last_timing = np.max(anode_trgs, axis=1)
            # last_timing[last_timing < 0] = 0
            max_car_img.add_values_to_image(max_anode_vals, cathode_amp)

            for n in np.arange(1, self.n_anodes + 1):  # n is channel id
                idx = n - 1  # python indexes at 0
                ch_max_idx = np.argwhere(np.argmax(corrected, axis=0) == idx)  # i.e. where ch is max
                ch_max_vals = corrected[idx, ch_max_idx].flatten()
                ch_max_timing = cathode_amp[ch_max_idx].flatten()
                anode_car_imgs[idx].add_values_to_image(ch_max_vals, ch_max_timing)

            print("{x} events read!".format(x=curr_idx + self.block_read_evts))
            curr_idx += self.block_read_evts

        print("All events read!")
        print("Max hist sum: ", max_car_img.img.sum())

        self.max_anode_cathode_histogram, self.anode_cathode_histograms = max_car_img, anode_car_imgs

    def plot_2d_ca_histograms(self, **kwargs):
        # self._create_timing_2d_histograms(**kwargs)
        max_hist, anode_histograms = self.max_anode_cathode_histogram, self.anode_cathode_histograms
        # energy_limits=None, time_limits=None, time_bins=1000
        a_edges, c_edges = max_hist.bins

        a_extent = (a_edges[0], a_edges[-1])
        c_extent = (c_edges[0], c_edges[-1])
        # y_extent = (t_edges[0], t_edges[-1])

        print("Anode extent: ", a_extent)
        print("Cathode extent: ", c_extent)

        nrow = self.n_anodes // 4
        ncol = self.n_anodes // 4

        fig1, axs1 = plt.subplots(nrow, ncol, figsize=(16, 12))
        for i, (ax, anode_hist) in enumerate(zip(fig1.axes, anode_histograms)):
            ch_img = anode_hist.img
            ch_img[0, :] = 0
            img = ax.imshow(ch_img, cmap='magma_r', origin='lower', interpolation='none', extent=np.append(a_extent, c_extent), aspect='auto')
            fig1.colorbar(img, fraction=0.046, pad=0.04, ax=ax)
            ax.set_xlabel('Ch ' + str(i + 1) + ' ADC bin')
            ax.set_ylabel('Cathode ADC bin')
            ax.set_xlim((0, 1000))
            ax.set_ylim(150, 350)

        fig2, ax2 = plt.subplots(1, 1, figsize=(6, 6))
        m_img = max_hist.img
        print("Max Hist filtered (percent):", m_img[0, :].sum() / m_img.sum())
        m_img[0, :] = 0
        img = ax2.imshow(m_img, cmap='magma_r', origin='lower', interpolation='none', extent=np.append(a_extent, c_extent), aspect='auto')
        fig2.colorbar(img, fraction=0.046, pad=0.04, ax=ax2)
        ax2.set_xlim((0, 1000))
        ax2.set_ylim(150, 350)
        ax2.set_xlabel('Max Anode ADC bin')
        ax2.set_ylabel('Cathode ADC bin')

        fig1.tight_layout()
        fig2.tight_layout()

        plt.show()

    def plot_1D_projections(self, anode_log_scale=True, cathode_log_scale=False,
                            **kwargs):
        # self._create_timing_2d_histograms(**kwargs)
        max_hist, anode_histograms = self.max_anode_cathode_histogram, self.anode_cathode_histograms
        # return self.a_edges, self.c_edges  # anode, cathode
        a_edges, c_edges = max_hist.bins

        print("Max hist shape: ", max_hist.img.shape)
        # print("max hist sum ax 0: ", np.sum(max_hist.img, axis=0).shape)
        a_step_bins = (a_edges[1:] + a_edges[:-1])/2
        c_step_bins = (c_edges[1:] + c_edges[:-1])/2

        print("en_steps: ", a_step_bins.size)
        print("t steps: ", c_step_bins.size)

        nrow = self.n_anodes // 4
        ncol = self.n_anodes // 4

        fig1, axs1 = plt.subplots(nrow, ncol, figsize=(16, 12))
        fig2, axs2 = plt.subplots(nrow, ncol, figsize=(16, 12))

        for i, (a_ax, c_ax, ch_obj) in enumerate(zip(fig1.axes, fig2.axes, anode_histograms)):  # energy

            ch_hist = ch_obj.img
            # ch_hist[0, :] = 0

            a_proj = np.sum(ch_hist, axis=0)  # anode projection
            c_proj = np.sum(ch_hist, axis=1)  # cathode projection

            a_ax.step(a_step_bins, a_proj, 'b-', where='mid')
            c_ax.step(c_step_bins, c_proj, 'b-', where='mid')

            a_ax.set_title("Amplitude (Channel {n})".format(n=i + 1))
            c_ax.set_title("Amplitude (Channel {n})".format(n=i + 1))
            # ax.set_xlabel('ch ' + str(i+1))
            a_ax.set_ylabel('Counts')
            c_ax.set_ylabel('Counts')

            if anode_log_scale:
                a_ax.set_yscale('log')

            if cathode_log_scale:
                c_ax.set_yscale('log')

        fig1.tight_layout()
        fig2.tight_layout()

        fig3, (ax31, ax32) = plt.subplots(1, 2, figsize=(16, 12))  # ax31 -> energy, ax32 -> time
        max_img = max_hist.img
        # max_img[0, :] = 0
        a_max_proj = np.sum(max_img, axis=0)  # anode max ch projection
        c_max_proj = np.sum(max_img, axis=1)  # anode max ch projection

        ax31.step(a_step_bins, a_max_proj, 'b-', where='mid')
        ax32.step(c_step_bins, c_max_proj, 'b-', where='mid')

        ax31.set_ylabel('Counts')
        ax32.set_ylabel('Counts')

        ax31.set_xlabel('Anode ADC')
        ax32.set_xlabel('Cathode ADC')

        if anode_log_scale:
            ax31.set_yscale('log')  # energy

        if cathode_log_scale:
            ax32.set_yscale('log')  # time

        # ax32.set_xlim([35, 100])
        fig3.tight_layout()

        plt.show()

    def cathode_gating(self, ch, low=210, high=250, log_scale=False, fit_gaussian=True, fit_cs137=True):
        # Mostly to give to hadong. Plot 1D original spectrum and cathode height gated spectrum.
        anode_histograms = self.anode_cathode_histograms[ch-1]  # 0 offset in python
        # return self.a_edges, self.c_edges  # anode, cathode
        a_edges, c_edges = anode_histograms.bins
        img = anode_histograms.img

        a_step_bins = (a_edges[1:] + a_edges[:-1]) / 2
        c_step_bins = (c_edges[1:] + c_edges[:-1]) / 2

        c_gated_idx = (c_step_bins < high) & (c_step_bins > low)

        fig, ax = plt.subplots(1, 1)

        a_proj_unfiltered = np.sum(img, axis=0)  # anode projection unfiltered
        a_proj_filtered = np.sum(img[c_gated_idx, :], axis=0)  # anode projection cathode height filtered

        ax.step(a_step_bins, a_proj_unfiltered, 'b-', where='mid', label='ungated')
        ax.step(a_step_bins, a_proj_filtered, 'r-', where='mid', label='gated')

        ax.set_ylabel('Counts')
        ax.set_xlim((0, 1000))

        if log_scale:
            ax.set_yscale('log')

        if fit_gaussian:
            if fit_cs137:
                windowed_data = a_proj_filtered.copy()
                windowed_data[:int(0.95 * np.argmax(a_proj_filtered))] = 0
                popt, pcov = gft.gauss_fit(a_step_bins, windowed_data)
                fit_amplitude, fit_center, fit_sigma, fit_bkg = popt
                resolution = (2.355 * fit_sigma)/fit_center
                print("Resolution: ", resolution)
                ax.step(a_step_bins, gft.gauss(a_step_bins, *popt), 'g--', label='fit: {:.2%} resolution'.format(resolution))

            else:  # TODO: general case
                pass

            ax.set_xlabel("Channel {n} ADC Bin".format(n=ch))
        else:
            ax.set_title("Amplitude (Channel {n})".format(n=ch))

        ax.legend(loc='best')
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


class CAImage(object):
    def __init__(self, ranges, bins):
        """X axis is energy, y axis is time, range=[energy_range, time_range], bins=(energy_bins, time_bins))"""
        img_hist, self.a_edges, self.c_edges = np.histogram2d([], [], range=ranges, bins=bins)
        self._img_hist = img_hist.T  # Transpose needed

    def add_values_to_image(self, energy_vals, time_vals):
        img_hist = np.histogram2d(energy_vals, time_vals, bins=[self.a_edges, self.c_edges])[0]
        self._img_hist += img_hist.T

    @property
    def img(self):
        return self._img_hist.copy()

    @property
    def bins(self):
        return self.a_edges, self.c_edges  # anode, cathode


def main(filename, **kwargs):
    prc = Analysis(filename, **kwargs)
    # prc.det_calibration = 540 / np.array([540, 540, 540, 540, 570, 540, 530, 540, 550, 560, 550, 550, 270, 550, 560, 540])
    prc.det_calibration = np.array([1, 0.66, 1, 1,
                                    1, 1, 1, 1,
                                    1, 1, 1, 1,
                                    2, 1, 1, 1])
    prc.create_ca_2d_histograms()
    prc.plot_2d_ca_histograms()
    prc.plot_1D_projections()
    prc.cathode_gating(3)

    prc.close()


if __name__ == "__main__":
    import os
    from pathlib import Path

    # data_file_name = "DavisD2022_9_23T15_51_clean.h5"  # weekend run
    data_file_name = "DavisD2022_10_20T12_7_clean.h5"  # Cs137, good data set
    fname = os.path.join(str(Path(os.getcwd()).parents[1]), "sample_data", "Data_hdf5", data_file_name)
    print("fname: ", fname)

    main(fname)
