import numpy as np
import unittest
import os
import tempfile
# import matplotlib.pyplot as plt
from pystripe import core


class TestWavedec(unittest.TestCase):
    def test(self):
        img = np.eye(5)
        coeffs = core.wavedec(img, wavelet='db1', level=None)
        approx = coeffs[0]
        self.assertEqual(len(coeffs), 3)
        self.assertTrue(np.allclose(approx, np.array([[1, 0], [0, 4]])))


class TestWaverec(unittest.TestCase):
    def test(self):
        img = np.eye(6)
        wavelet = 'db1'
        coeffs = core.wavedec(img, wavelet=wavelet, level=None)
        recon = core.waverec(coeffs, wavelet=wavelet)
        self.assertTrue(np.allclose(img, recon))


# def plot_fft(data, fdata):
#     plt.subplot(121)
#     plt.imshow(data)
#     plt.subplot(122)
#     plt.imshow(np.sqrt(np.real(fdata) ** 2 + np.imag(fdata) ** 2))
#     plt.show()


class TestFFT(unittest.TestCase):
    def setUp(self):
        self.data = np.zeros((64, 64))
        self.data[12, :] = 10  # thin horizontal stripe, should show up as high frequency vertical component

    def test_shift(self):
        fdata = core.fft(self.data)
        self.assertAlmostEqual(fdata[44, 32], 640.0)

    def test_noshift(self):
        fdata = core.fft(self.data, shift=False)
        self.assertAlmostEqual(fdata[12, 0], 640.0)


class TestFFT2(unittest.TestCase):
    def setUp(self):
        self.data = np.zeros((64, 64))
        self.data[12, :] = 10  # thin horizontal stripe, should show up as high frequency vertical component

    def test_shift(self):
        fdata = core.fft2(self.data)
        self.assertAlmostEqual(fdata[44, 32], np.complex(0, -640.0))


class TestNotch(unittest.TestCase):
    def test(self):
        g = core.notch(n=4, sigma=1)
        self.assertTrue(np.allclose(g, np.array([0, 0.39346934, 0.86466472, 0.988891])))
        g = core.notch(n=4, sigma=2)
        self.assertTrue(np.allclose(g, np.array([0, 0.1175031, 0.39346934, 0.67534753])))

    def test_zero(self):
        with self.assertRaises(ValueError):
            g = core.notch(n=4, sigma=0.0)
        with self.assertRaises(ValueError):
            g = core.notch(n=0, sigma=1.0)


class TestGaussianFilter(unittest.TestCase):
    def test(self):
        m = 10
        res = core.gaussian_filter(shape=(m, 4), sigma=1)
        self.assertTrue(np.allclose(res, np.array(m*[[0, 0.39346934, 0.86466472, 0.988891]])))
        res = core.gaussian_filter(shape=(m, 4), sigma=2)
        self.assertTrue(np.allclose(res, np.array(m*[[0, 0.1175031, 0.39346934, 0.67534753]])))


class TestFilterStreaks(unittest.TestCase):
    def test_odd(self):
        #
        # Bug - images with odd dimensions did not work because fft got
        #       padded to even dims
        #
        for shape in ((1000, 1001), (1001, 1000), (1001, 1001)):
            img = np.zeros(shape)
            result = core.filter_streaks(img, (128, 512), level=0, wavelet="db5", crossover=10)
            self.assertSequenceEqual(img.shape, result.shape)

if __name__ == '__main__':
    unittest.main()