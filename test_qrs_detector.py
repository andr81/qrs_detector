import unittest
import numpy as np
from QRSDetectorOffline import QRSDetectorOffline

class TestQRSDetector(unittest.TestCase):

    def assertEqualWithDelta(self, a, b, delta=0.0, msg=''):
        if not b >= a - delta or not b <= a + delta:
            msg += ': a={}, b={}, delta={}'.format(a, b, delta)
            raise self.failureException(msg)

    def assert_all_peaks(self, peaks_expected, peaks_actual, delta):
        self.assertEqual(len(peaks_expected), len(peaks_actual))
        for i in range(len(peaks_expected)):
            r_exp = peaks_expected[i]
            r_act = peaks_actual[i]
            self.assertEqualWithDelta(r_exp, r_act, delta, 'peak #{}'.format(i))

    def assert_peaks(self, peaks_expected, peaks_actual, delta):
        for i in range(len(peaks_expected)):
            r_exp = peaks_expected[i]
            nearest = np.extract((peaks_actual >= r_exp - delta) & (peaks_actual <= r_exp + delta), peaks_actual)
            if len(nearest) == 0:
                raise self.failureException('not found peak {}'.format(r_exp))

    def test_qrs_1(self):
        qrs_detector = QRSDetectorOffline(ecg_data_path="ecg_data/ecg_data_1.csv", verbose=False,
                                  log_data=False, plot_data=False, show_plot=False)
        expected_r = [194, 390, 581, 768, 955, 1142, 1329, 1515, 1701, 1886, 2070, 2253, 2434, 2616]
        self.assert_all_peaks(expected_r, qrs_detector.qrs_peaks_indices, 10)

        qrs_detector = QRSDetectorOffline(ecg_data_path="ecg_data/ecg_data_2.csv", verbose=False,
                                  log_data=False, plot_data=False, show_plot=False)
        expected_r = [84, 282, 483, 683, 874]
        self.assert_all_peaks(expected_r, qrs_detector.qrs_peaks_indices, 10)

    def test_qrs_2(self):
        qrs_detector = QRSDetectorOffline(ecg_data_path="ecg_data/ecg_data_1.csv", verbose=False,
                                  log_data=False, plot_data=False, show_plot=False, show_rs_points=True)
        expected_r = [194, 390, 581, 768, 955, 1142, 1329, 1515, 1701, 1886, 2070, 2253, 2434, 2616]
        self.assert_all_peaks(expected_r, qrs_detector.qrs_peaks_indices, 10)
        self.assertEqualWithDelta(82, qrs_detector.hr, 2, "HR")

        qrs_detector = QRSDetectorOffline(ecg_data_path="ecg_data/ecg_data_2.csv", verbose=False,
                                  log_data=False, plot_data=False, show_plot=False, show_rs_points=True)
        expected_r = [84, 282, 483, 683, 874]
        self.assert_all_peaks(expected_r, qrs_detector.qrs_peaks_indices, 10)
        self.assertEqualWithDelta(75, qrs_detector.hr, 2, "HR")

    def test_qrs_3(self):
        signal = np.loadtxt('ecg_data/test_3.txt')
        ecg_data_raw = np.array([[0,i] for i in signal[:]])
        qrs_detector = QRSDetectorOffline(ecg_data_path="", verbose=False,
                                        log_data=False, plot_data=False, show_plot=False,
                                        ecg_data_raw=ecg_data_raw, bps=500, findpeaks_limit=0.001, show_rs_points=True)
        self.assertEqualWithDelta(68, qrs_detector.hr, 1, "HR")
        self.assertEqualWithDelta(41.5, qrs_detector.sdnn, 1, "SDNN")
        expected_r = [273, 724, 1181, 1635, 2076, 2535, 2970, 3426, 3872, 4322, 4765, 5217, 5669, 6129, 6573, 7038, 7476, 7901, 8300, 
                    8675, 9077, 9471, 9899, 10323, 10726, 11163, 11594, 12054, 12494, 12927, 13361, 13804, 14240, 14686]
        self.assert_all_peaks(expected_r, np.array([x.r_index for x in qrs_detector.rs_complexes]), 20)

    def test_qrs_4(self):
        signal = np.loadtxt('ecg_data/test_4.txt')
        ecg_data_raw = np.array([[0,i] for i in signal[:]])
        qrs_detector = QRSDetectorOffline(ecg_data_path="", verbose=False,
                                        log_data=False, plot_data=False, show_plot=False,
                                        ecg_data_raw=ecg_data_raw, bps=500, findpeaks_limit=0.001, show_rs_points=True)
        self.assertEqualWithDelta(69, qrs_detector.hr, 1, "HR")
        self.assertEqualWithDelta(85, qrs_detector.sdnn, 3, "SDNN")
        expected_r = [976, 1462, 1917, 2366, 5508, 5944, 6410, 6861, 7328, 7786, 8234, 8674, 9122, 
                        9554, 9985, 10397, 10780, 11202, 11591, 11978, 12340, 12697, 13082, 13482, 13901, 14323]
        self.assert_peaks(expected_r, np.array([x.r_index for x in qrs_detector.rs_complexes]), 10)

    def test_hrv(self):
        signal = np.loadtxt('ecg_data/test_hrv_1.txt')
        ecg_data_raw = np.array([[0,i] for i in signal[:]])
        qrs_detector = QRSDetectorOffline(ecg_data_path="", verbose=False,
                                        log_data=False, plot_data=False, show_plot=False,
                                        ecg_data_raw=ecg_data_raw, bps=500, findpeaks_limit=0.001, show_rs_points=True)
        self.assertEqualWithDelta(92, qrs_detector.hr, 1, "HR")
        self.assertEqualWithDelta(55, qrs_detector.sdnn, 3, "SDNN")

if __name__ == '__main__':
    unittest.main()