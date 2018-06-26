import numpy as np
import matplotlib.pyplot as plt
from time import gmtime, strftime
from scipy.signal import butter, lfilter


LOG_DIR = "logs/"
PLOT_DIR = "plots/"

class RSComplex(object):
    r_index = None
    s_index = None
    r_diff_index = None
    s_diff_index = None
    r_amplitude = None
    s_amplitude = None
    r_diff_amplitude = None
    s_diff_amplitude = None

class QRSDetectorOffline(object):
    """
    Python Offline ECG QRS Detector based on the Pan-Tomkins algorithm.
    
    Michał Sznajder (Jagiellonian University) - technical contact (msznajder@gmail.com)
    Marta Łukowska (Jagiellonian University)


    The module is offline Python implementation of QRS complex detection in the ECG signal based
    on the Pan-Tomkins algorithm: Pan J, Tompkins W.J., A real-time QRS detection algorithm,
    IEEE Transactions on Biomedical Engineering, Vol. BME-32, No. 3, March 1985, pp. 230-236.

    The QRS complex corresponds to the depolarization of the right and left ventricles of the human heart. It is the most visually obvious part of the ECG signal. QRS complex detection is essential for time-domain ECG signal analyses, namely heart rate variability. It makes it possible to compute inter-beat interval (RR interval) values that correspond to the time between two consecutive R peaks. Thus, a QRS complex detector is an ECG-based heart contraction detector.

    Offline version detects QRS complexes in a pre-recorded ECG signal dataset (e.g. stored in .csv format).

    This implementation of a QRS Complex Detector is by no means a certified medical tool and should not be used in health monitoring. It was created and used for experimental purposes in psychophysiology and psychology.

    You can find more information in module documentation:
    https://github.com/c-labpl/qrs_detector

    If you use these modules in a research project, please consider citing it:
    https://zenodo.org/record/583770

    If you use these modules in any other project, please refer to MIT open-source license.


    MIT License

    Copyright (c) 2017 Michał Sznajder, Marta Łukowska

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
    """

    def __init__(self, ecg_data_path, verbose=True, log_data=False, plot_data=False, show_plot=False, 
                ecg_data_raw=None, bps=250, findpeaks_limit=0.35, show_rs_points=False):
        """
        QRSDetectorOffline class initialisation method.
        :param string ecg_data_path: path to the ECG dataset
        :param bool verbose: flag for printing the results
        :param bool log_data: flag for logging the results
        :param bool plot_data: flag for plotting the results to a file
        :param bool show_plot: flag for showing generated results plot - will not show anything if plot is not generated
        :param array ecg_data_raw: raw ecg data, filename will be ignored
        :param int bps: signal frequency, beats per secons
        :param bool show_rs_points: flag for detect and plot R/S points
        """
        # Configuration parameters.
        self.ecg_data_path = ecg_data_path
        self.show_rs_points = show_rs_points

        self.signal_frequency = int(bps)  # Set ECG device frequency in samples per second here.

        self.filter_lowcut = 0.1
        self.filter_highcut = 15.0
        self.filter_order = 1

        self.integration_window = int(15 * (bps / 250))  # Change proportionally when adjusting frequency (in samples).

        self.findpeaks_limit = findpeaks_limit
        self.findpeaks_spacing = int(50 * (bps / 250))  # Change proportionally when adjusting frequency (in samples).

        self.refractory_period = int(120 * (bps / 250))  # Change proportionally when adjusting frequency (in samples).
        self.qrs_peak_filtering_factor = 0.125
        self.noise_peak_filtering_factor = 0.125
        self.qrs_noise_diff_weight = 0.25

        # Loaded ECG data.
        self.ecg_data_raw = ecg_data_raw

        # Measured and calculated values.
        self.filtered_ecg_measurements = None
        self.differentiated_ecg_measurements = None
        self.squared_ecg_measurements = None
        self.integrated_ecg_measurements = None
        self.detected_peaks_indices = None
        self.detected_peaks_values = None

        self.qrs_peak_value = 0.0
        self.noise_peak_value = 0.0
        self.threshold_value = 0.0

        # Detection results.
        self.qrs_peaks_indices = np.array([], dtype=int)
        self.noise_peaks_indices = np.array([], dtype=int)
        
        # R/S poins detection results
        self.rs_complexes = np.array([], dtype=RSComplex)

        # heart rate and variability
        self.hrv = None
        self.hr = None

        # Final ECG data and QRS detection results array - samples with detected QRS are marked with 1 value.
        self.ecg_data_detected = None

        # Run whole detector flow.
        if ecg_data_raw is None:
            self.load_ecg_data()
        self.detect_peaks()
        self.detect_qrs()
        if show_rs_points:
            self.detect_rs_points()

        if verbose:
            self.print_detection_data()

        if log_data:
            self.log_path = "{:s}QRS_offline_detector_log_{:s}.csv".format(LOG_DIR,
                                                                           strftime("%Y_%m_%d_%H_%M_%S", gmtime()))
            self.log_detection_data()

        if plot_data:
            self.plot_path = "{:s}QRS_offline_detector_plot_{:s}.png".format(PLOT_DIR,
                                                                             strftime("%Y_%m_%d_%H_%M_%S", gmtime()))
            self.plot_detection_data(show_plot=show_plot)

    """Loading ECG measurements data methods."""

    def load_ecg_data(self):
        """
        Method loading ECG data set from a file.
        """
        self.ecg_data_raw = np.loadtxt(self.ecg_data_path, skiprows=1, delimiter=',')

    """ECG measurements data processing methods."""

    def detect_peaks(self):
        """
        Method responsible for extracting peaks from loaded ECG measurements data through measurements processing.
        """
        # Extract measurements from loaded ECG data.
        ecg_measurements = self.ecg_data_raw[:, 1]

        # Measurements filtering - 0-15 Hz band pass filter.
        self.filtered_ecg_measurements = self.bandpass_filter(ecg_measurements, lowcut=self.filter_lowcut,
                                                              highcut=self.filter_highcut, signal_freq=self.signal_frequency,
                                                              filter_order=self.filter_order)
        self.filtered_ecg_measurements[:5] = self.filtered_ecg_measurements[5]

        # Derivative - provides QRS slope information.
        self.differentiated_ecg_measurements = np.ediff1d(self.filtered_ecg_measurements)

        # Squaring - intensifies values received in derivative.
        self.squared_ecg_measurements = self.differentiated_ecg_measurements ** 2

        # Moving-window integration.
        self.integrated_ecg_measurements = np.convolve(self.squared_ecg_measurements, np.ones(self.integration_window))

        # Fiducial mark - peak detection on integrated measurements.
        self.detected_peaks_indices = self.findpeaks(data=self.integrated_ecg_measurements,
                                                     limit=self.findpeaks_limit,
                                                     spacing=self.findpeaks_spacing)

        self.detected_peaks_values = self.integrated_ecg_measurements[self.detected_peaks_indices]

    """QRS detection methods."""

    def detect_qrs(self):
        """
        Method responsible for classifying detected ECG measurements peaks either as noise or as QRS complex (heart beat).
        """
        for detected_peak_index, detected_peaks_value in zip(self.detected_peaks_indices, self.detected_peaks_values):

            try:
                last_qrs_index = self.qrs_peaks_indices[-1]
            except IndexError:
                last_qrs_index = 0

            # After a valid QRS complex detection, there is a 200 ms refractory period before next one can be detected.
            if detected_peak_index - last_qrs_index > self.refractory_period or not self.qrs_peaks_indices.size:
                # Peak must be classified either as a noise peak or a QRS peak.
                # To be classified as a QRS peak it must exceed dynamically set threshold value.
                if detected_peaks_value > self.threshold_value:
                    self.qrs_peaks_indices = np.append(self.qrs_peaks_indices, detected_peak_index)

                    # Adjust QRS peak value used later for setting QRS-noise threshold.
                    self.qrs_peak_value = self.qrs_peak_filtering_factor * detected_peaks_value + \
                                          (1 - self.qrs_peak_filtering_factor) * self.qrs_peak_value
                else:
                    self.noise_peaks_indices = np.append(self.noise_peaks_indices, detected_peak_index)

                    # Adjust noise peak value used later for setting QRS-noise threshold.
                    self.noise_peak_value = self.noise_peak_filtering_factor * detected_peaks_value + \
                                            (1 - self.noise_peak_filtering_factor) * self.noise_peak_value

                # Adjust QRS-noise threshold value based on previously detected QRS or noise peaks value.
                self.threshold_value = self.noise_peak_value + \
                                       self.qrs_noise_diff_weight * (self.qrs_peak_value - self.noise_peak_value)

        # Create array containing both input ECG measurements data and QRS detection indication column.
        # We mark QRS detection with '1' flag in 'qrs_detected' log column ('0' otherwise).
        measurement_qrs_detection_flag = np.zeros([len(self.ecg_data_raw[:, 1]), 1])
        measurement_qrs_detection_flag[self.qrs_peaks_indices] = 1
        self.ecg_data_detected = np.append(self.ecg_data_raw, measurement_qrs_detection_flag, 1)

    """Results reporting methods."""

    def print_detection_data(self):
        """
        Method responsible for printing the results.
        """
        print("qrs peaks indices")
        print(self.qrs_peaks_indices)
        print("noise peaks indices")
        print(self.noise_peaks_indices)

    def log_detection_data(self):
        """
        Method responsible for logging measured ECG and detection results to a file.
        """
        with open(self.log_path, "wb") as fin:
            fin.write(b"timestamp,ecg_measurement,qrs_detected\n")
            np.savetxt(fin, self.ecg_data_detected, delimiter=",")

    def plot_detection_data(self, show_plot=False):
        """
        Method responsible for plotting detection results.
        :param bool show_plot: flag for plotting the results and showing plot
        """
        def plot_data(axis, data, title='', fontsize=10):
            axis.set_title(title, fontsize=fontsize)
            axis.grid(which='both', axis='both', linestyle='--')
            axis.plot(data, color="salmon", zorder=1)

        def plot_points(axis, values, indices, c="black"):
            axis.scatter(x=indices, y=values[indices], c=c, s=50, zorder=2)

        plt.close('all')
        fig, axarr = plt.subplots(6, sharex=True, figsize=(15, 18))

        data = self.ecg_data_detected[:, 1]

        plot_data(axis=axarr[0], data=data, title='Raw ECG measurements')
        plot_data(axis=axarr[1], data=self.filtered_ecg_measurements, title='Filtered ECG measurements')
        plot_data(axis=axarr[2], data=self.differentiated_ecg_measurements, title='Differentiated ECG measurements')
        plot_data(axis=axarr[3], data=self.squared_ecg_measurements, title='Squared ECG measurements')
        plot_data(axis=axarr[4], data=self.integrated_ecg_measurements, title='Integrated ECG measurements with QRS peaks marked (black)')
        plot_points(axis=axarr[4], values=self.integrated_ecg_measurements, indices=self.qrs_peaks_indices)
        plot_data(axis=axarr[5], data=self.ecg_data_detected[:, 1], title='Raw ECG measurements with QRS peaks marked (black)')

        if self.show_rs_points:
            # raw
            r_point_indices = [rs.r_index for rs in self.rs_complexes]
            s_point_indices = [rs.s_index for rs in self.rs_complexes]
            r_amplitudes = [rs.r_amplitude for rs in self.rs_complexes]
            s_amplitudes = [rs.s_amplitude for rs in self.rs_complexes]
            plot_points(axis=axarr[5], values=data, indices=r_point_indices, c='red')
            plot_points(axis=axarr[5], values=data, indices=s_point_indices, c='green')
            axarr[5].vlines(x=r_point_indices, ymin=data[r_point_indices] - r_amplitudes, 
                        ymax=data[r_point_indices], colors='red', zorder=2)
            axarr[5].vlines(x=s_point_indices, ymin=data[s_point_indices], 
                        ymax=data[s_point_indices] + s_amplitudes, colors='green', zorder=2)

            # diff
            r_diff_amplitudes = [rs.r_diff_amplitude for rs in self.rs_complexes]
            s_diff_amplitudes = [rs.s_diff_amplitude for rs in self.rs_complexes]
            r_diff_point_indices = [rs.r_diff_index for rs in self.rs_complexes]
            s_diff_point_indices = [rs.s_diff_index for rs in self.rs_complexes]
            plot_points(axis=axarr[2], values=self.differentiated_ecg_measurements, indices=r_diff_point_indices, c='red')
            plot_points(axis=axarr[2], values=self.differentiated_ecg_measurements, indices=s_diff_point_indices, c='green')
            axarr[2].vlines(x=r_diff_point_indices, 
                        ymin=self.differentiated_ecg_measurements[r_diff_point_indices] - r_diff_amplitudes, 
                        ymax=self.differentiated_ecg_measurements[r_diff_point_indices], 
                        colors='red', zorder=2)
            axarr[2].vlines(x=s_diff_point_indices, 
                        ymin=self.differentiated_ecg_measurements[s_diff_point_indices], 
                        ymax=self.differentiated_ecg_measurements[s_diff_point_indices] + s_diff_amplitudes, 
                        colors='green', zorder=2)
        else:
            plot_points(axis=axarr[5], values=self.ecg_data_detected[:, 1], indices=self.qrs_peaks_indices)

        plt.tight_layout()
        fig.savefig(self.plot_path)

        if show_plot:
            plt.show()

        plt.close()

    """Tools methods."""

    def bandpass_filter(self, data, lowcut, highcut, signal_freq, filter_order):
        """
        Method responsible for creating and applying Butterworth filter.
        :param deque data: raw data
        :param float lowcut: filter lowcut frequency value
        :param float highcut: filter highcut frequency value
        :param int signal_freq: signal frequency in samples per second (Hz)
        :param int filter_order: filter order
        :return array: filtered data
        """
        nyquist_freq = 0.5 * signal_freq
        low = lowcut / nyquist_freq
        high = highcut / nyquist_freq
        b, a = butter(filter_order, [low, high], btype="band")
        y = lfilter(b, a, data)
        return y

    def findpeaks(self, data, spacing=1, limit=None):
        """
        Janko Slavic peak detection algorithm and implementation.
        https://github.com/jankoslavic/py-tools/tree/master/findpeaks
        Finds peaks in `data` which are of `spacing` width and >=`limit`.
        :param ndarray data: data
        :param float spacing: minimum spacing to the next peak (should be 1 or more)
        :param float limit: peaks should have value greater or equal
        :return array: detected peaks indexes array
        """
        len = data.size
        x = np.zeros(len + 2 * spacing)
        x[:spacing] = data[0] - 1.e-6
        x[-spacing:] = data[-1] - 1.e-6
        x[spacing:spacing + len] = data
        peak_candidate = np.zeros(len)
        peak_candidate[:] = True
        for s in range(spacing):
            start = spacing - s - 1
            h_b = x[start: start + len]  # before
            start = spacing
            h_c = x[start: start + len]  # central
            start = spacing + s + 1
            h_a = x[start: start + len]  # after
            peak_candidate = np.logical_and(peak_candidate, np.logical_and(h_c > h_b, h_c > h_a))

        ind = np.argwhere(peak_candidate)
        ind = ind.reshape(ind.size)
        if limit is not None:
            ind = ind[data[ind] > limit]
        return ind

    def detect_rs_points(self):
        """
        Detection of R/S points as a local maximum and minimum of a differentiated curve
        """

        def get_local_max_ind(arr, pos1, pos2):
            # the maximum point in the window
            win_start = max(0, pos1)
            win = arr[win_start: pos2]
            mx = np.argmax(win)
            return mx + win_start

        def get_local_min_ind(arr, pos1, pos2):
            # the minimum point in the window
            win_start = max(0, pos1)
            win = arr[win_start: pos2]
            mx = np.argmin(win)
            return mx + win_start

        def get_amplitudes(r_point, s_point, data, rr_mean):
            # R/S/ peaks amplitudes
            r_value = data[r_point]
            s_value = data[s_point]
            win_size = int(rr_mean / 4)
            win_start = max(0, r_point - win_size)
            win = data[win_start: r_point]
            # median before R-peak as zero level
            zero = np.median(win)
            return max(0, r_value - zero), max(0, zero - s_value)

        if self.qrs_peaks_indices.shape[0] < 4:
            return

        for peak_ind in self.qrs_peaks_indices:
            if peak_ind < self.findpeaks_spacing or peak_ind > self.ecg_data_raw.shape[0] -  self.findpeaks_spacing:
                continue
            rs_complex = RSComplex()
            # max/min from diff
            rs_complex.s_diff_index = get_local_min_ind(self.differentiated_ecg_measurements, 
                                    peak_ind - self.findpeaks_spacing * 2, peak_ind + self.findpeaks_spacing * 2)
            rs_complex.r_diff_index = get_local_max_ind(self.differentiated_ecg_measurements, 
                                    peak_ind - self.findpeaks_spacing * 2, peak_ind + self.findpeaks_spacing * 2)
            # max diff point at the left of S (to find real R-peak)
            r_ind = get_local_max_ind(self.differentiated_ecg_measurements, 
                                    peak_ind - self.findpeaks_spacing * 2, rs_complex.s_diff_index)
            # clarify by raw data
            rs_complex.r_index = get_local_max_ind(self.ecg_data_raw[:,1], 
                            r_ind - int(self.findpeaks_spacing/10), r_ind + int(self.findpeaks_spacing/5))
            rs_complex.s_index = get_local_min_ind(self.ecg_data_raw[:,1], 
                            rs_complex.s_diff_index - int(self.findpeaks_spacing/5), rs_complex.s_diff_index + int(self.findpeaks_spacing/5))

            self.rs_complexes = np.append(self.rs_complexes, rs_complex)

        # heart rate and variability
        r_point_indices = [rs.r_index for rs in self.rs_complexes]
        rr_intervals = [r_point_indices[i] - r_point_indices[i-1] 
                        for i in range(1, len(r_point_indices))]
        self.hrv = int(round(np.std(rr_intervals) * self.signal_frequency / 1000))
        rr_mean = np.median(rr_intervals) * self.signal_frequency / 1000
        self.hr = int(round(60 / rr_mean * 1000))

        # find zero level for every qrs-complex, calculate R/S-amplitudes
        for rs_complex in self.rs_complexes:
            rs_complex.r_amplitude, rs_complex.s_amplitude = get_amplitudes(rs_complex.r_index, rs_complex.s_index, 
                                                    self.ecg_data_raw[:, 1], rr_mean)
            rs_complex.r_diff_amplitude, rs_complex.s_diff_amplitude = get_amplitudes(rs_complex.r_diff_index, rs_complex.s_diff_index, 
                                                    self.differentiated_ecg_measurements, rr_mean)

# import pyedflib 

if __name__ == "__main__":
    qrs_detector = QRSDetectorOffline(ecg_data_path="ecg_data/ecg_data_1.csv", verbose=True,
                                      log_data=True, plot_data=False, show_plot=False)

    # f = pyedflib.EdfReader('/Users/vita/Downloads/Lip/2018-05-31_10.54.06.edf')
    # sig = f.readSignal(f.getSignalLabels().index('ECG_V2'))[0:10000]
    # f._close()
    # ecg_data_raw = np.array([[0,i] for i in sig])
    # qrs_detector = QRSDetectorOffline(ecg_data_path="", verbose=True,
    #                               log_data=True, plot_data=True, 
    #                               show_plot=True,
    #                               ecg_data_raw=ecg_data_raw, bps = 1000, findpeaks_limit=0.001, show_rs_points=True)
