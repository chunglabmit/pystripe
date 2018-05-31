import pystripe
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
import pywt
from skimage import filters
from skimage.morphology import disk
from skimage.transform import resize


project_folder = '/media/jswaney/Drive1/Justin/destripe/125400_100600'


def gradient(data):
    fy, fx = np.gradient(data)
    grad = np.zeros((*data.shape, 2))
    grad[:, :, 0] = fy
    grad[:, :, 1] = fx
    return grad


def hessian(data):
    grad = np.gradient(data)

    n_dims = len(grad)
    H = np.zeros((*data.shape, n_dims, n_dims))
    for i, first_deriv in enumerate(grad):
        for j in range(i, n_dims):
            second_deriv = np.gradient(first_deriv, axis=j)
            H[:, :, i, j] = second_deriv
            if i != j:
                H[:, :, j, i] = second_deriv
    return H


def decibel(data):
    return 20*np.log10(data+1e-6)


def new_method():
    img_path = 'crop.tif'
    wavelet = 'db2'
    level = None
    sigma = 1.0
    cmax = 300
    a = 1
    b = 1
    threshold = 0.2
    width_frac = 0.1

    img = pystripe.imread(img_path)
    img = np.array(img, dtype=np.float)

    g = filters.gaussian(img, sigma)
    g = img
    gy, gx = np.gradient(g)
    gyy, gyx = np.gradient(gy)

    h = hessian(g)
    eigvals = np.linalg.eigvalsh(h)

    f = np.exp(-eigvals[:, :, 0]**2/(2*a**2))*(1-np.exp(-np.clip(gyy, 0, None)**2/(2*b**2)))
    f = filters.median(f, disk(3))
    f = f / f.max()

    # mask = (f > threshold)

    ff = pystripe.fft2(f, shift=False)

    # maskf = pystripe.fft2(mask)
    # imgf = pystripe.fft2(g)

    filt = np.zeros(img.shape)
    loc = np.where(decibel(pystripe.magnitude(ff))>50)
    filt[loc] = 1
    filt = filters.gaussian(filt, 2)

    # blob = np.zeros(img.shape)
    # blob[0, 0] = 1
    # blob = filters.gaussian(blob, sigma=5)
    #
    # filt[np.where(blob>0.01)] = 0

    plt.imshow(decibel(pystripe.magnitude(ff)))
    plt.show()
    plt.imshow(filt)
    plt.show()
    plt.plot(decibel(pystripe.magnitude(ff)).max(axis=-1))
    plt.show()

    # coeffs = pystripe.wavedec(img, wavelet=wavelet, level=level)
    # approx = coeffs[0]
    # detail = coeffs[1:]
    # coeffs_filt = [approx]
    # for ch, cv, cd in detail:
    #     chf = pystripe.fft(ch, shift=False)
    #     m = resize(filt, ch.shape)
    #     chf_filt = chf * m
    #     ch_filt = pystripe.ifft(chf_filt)
    #     coeffs_filt.append((ch_filt, cv, cd))
    #
    # fimg = pystripe.waverec(coeffs_filt, wavelet)
    #
    # # scaled_fimg = pystripe.hist_match(fimg, img)
    # # np.clip(fimg, np.iinfo(img.dtype).min, np.iinfo(img.dtype).max, out=fimg)
    #
    # plt.subplot(121)
    # plt.imshow(img, clim=[0, cmax])
    # plt.subplot(122)
    # plt.imshow(fimg, clim=[0, cmax])
    # plt.show()





def wavelet_fft_method():
    img_path = '044780.raw'
    wavelet = 'db5'
    level = None
    sigma1 = 256
    sigma2 = 256
    cmax = 300
    threshold = None
    inflection = 100
    spread = 100

    img = pystripe.imread(img_path)
    img = np.array(img, dtype=np.float)

    if threshold is None:
        threshold = filters.threshold_otsu(img)/2
        inflection = threshold
        print(threshold)

    background = np.clip(img, None, threshold)
    foreground = np.clip(img, threshold, None)

    min_len = int(min(img.shape))
    max_level = pystripe.max_level(min_len, wavelet)

    print(f"Using {wavelet} wavelet on a {img.shape} {img.dtype} image")
    print(f"The maximum possible DWT level is {max_level}")
    # print(f"Filter bandwidth: {sigma1} pixels, {width_frac*100} percent of vertical dimension")

    coeffs = pystripe.wavedec(background, wavelet=wavelet, level=level)
    approx = coeffs[0]
    detail = coeffs[1:]
    coeffs_filt = [approx]
    width_frac = sigma1 / img.shape[0]
    for ch, cv, cd in detail:
        s = ch.shape[0] * width_frac
        chf = pystripe.fft(ch, shift=False)
        mask = pystripe.gaussian_filter(chf.shape, sigma=s)
        chf_filt = chf * mask
        ch_filt = pystripe.ifft(chf_filt)
        # ch_filt = filters.gaussian(ch_filt, sigma=10)
        coeffs_filt.append((ch_filt, cv, cd))
    background_filtered = pystripe.waverec(coeffs_filt, wavelet)

    coeffs = pystripe.wavedec(foreground, wavelet=wavelet, level=level)
    approx = coeffs[0]
    detail = coeffs[1:]
    coeffs_filt = [approx]
    width_frac = sigma2 / img.shape[0]
    for ch, cv, cd in detail:
        s = ch.shape[0] * width_frac
        chf = pystripe.fft(ch, shift=False)
        mask = pystripe.gaussian_filter(chf.shape, sigma=s)
        chf_filt = chf * mask
        ch_filt = pystripe.ifft(chf_filt)
        # ch_filt = filters.gaussian(ch_filt, sigma=10)
        coeffs_filt.append((ch_filt, cv, cd))
    foreground_filtered = pystripe.waverec(coeffs_filt, wavelet)

    f = pystripe.foreground_fraction(img, inflection, spread, smoothing=1)
    # f = filters.gaussian(f, sigma=1)
    fimg = foreground_filtered*f + background*(1-f)
    # fimg = np.maximum(fimg, img)

    plt.subplot(131)
    plt.imshow(img, clim=[0, cmax])
    plt.subplot(132)
    plt.imshow(background_filtered, clim=[0, cmax])
    plt.subplot(133)
    plt.imshow(fimg, clim=[0, cmax])
    plt.show()

    plt.imshow(img, clim=[0, 2 * cmax])
    plt.show()
    plt.imshow(fimg, clim=[0, 2*cmax])
    plt.show()


    # x = np.linspace(0, 4000)
    # y = pystripe.foreground_fraction(x, inflection, spread)
    # plt.plot(x, y)
    # plt.show()
    #
    # fimg = f*img + (1-f)*background_filtered
    #
    # # scaled_fimg = pystripe.hist_match(fimg, img)
    # # np.clip(fimg, np.iinfo(img.dtype).min, np.iinfo(img.dtype).max, out=fimg)
    # np.clip(fimg, 0, 4095, out=fimg)
    #
    # plt.subplot(121)
    # plt.imshow(img, clim=[0, cmax])
    # plt.subplot(122)
    # plt.imshow(f, clim=[0, cmax])
    # plt.show()


def main():
    wavelet_fft_method()
    # new_method()

    # img_path = 'crop.tif'
    # cmax = 300
    # sigma = 300

    # img = np.array(pystripe.imread(img_path), dtype=np.float)
    #
    # lpass = ndimage.gaussian_filter1d(img, sigma=sigma, axis=0)
    # hpass = img-lpass
    #
    #
    # plt.subplot(121)
    # plt.imshow(lpass, clim=[0, cmax])
    # plt.subplot(122)
    # plt.imshow(hpass)
    # plt.show()


if __name__ == '__main__':
    main()