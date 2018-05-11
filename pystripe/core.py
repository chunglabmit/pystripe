import argparse
from pathlib import Path
import os
import numpy as np
from scipy import fftpack
import tifffile
from tsv import raw
import pywt
import multiprocessing
import tqdm


supported_extensions = ['.tif', '.tiff', '.raw']


def _get_extension(path):
    """Extract the file extension from the provided path
    
    Parameters
    ----------
    path : str
        path with a file extension

    Returns
    -------
    ext : str
        file extension of provided path

    """
    return os.path.splitext(path)[1]


def imread(path):
    """Load a tiff or raw image
    
    Parameters
    ----------
    path : str
        path to tiff or raw image

    Returns
    -------
    img : ndarray
        image as a numpy array

    """
    img = None
    extension = _get_extension(path)
    if extension == '.raw':
        img = raw.raw_imread(path)
    elif extension == '.tif' or extension == '.tiff':
        img = tifffile.imread(path)
    return img


def imsave(path, img, compression=1):
    """Save an array as a tiff or raw image
    
    The file format will be inferred from the file extension in `path`
    
    Parameters
    ----------
    path : str
        path to tiff or raw image
    img : ndarray
        image as a numpy array
    compression : int
        compression level for tiff writing

    """
    extension = _get_extension(path)
    if extension == '.raw':
        # TODO: get raw writing to work
        # raw.raw_imsave(path, img)
        tifffile.imsave(os.path.splitext(path)[0]+'.tif', img, compress=compression)
    elif extension == '.tif' or extension == '.tiff':
        tifffile.imsave(path, img, compress=compression)


def wavedec(img, wavelet, level=None):
    """Decompose `img` using discrete (decimated) wavelet transform using `wavelet`
    
    Parameters
    ----------
    img : ndarray
        image to be decomposed into wavelet coefficients
    wavelet : str
        name of the mother wavelet
    level : int (optional)
        number of wavelet levels to use. Default is the maximum possible decimation

    Returns
    -------
    coeffs : list
        the approximation coefficients followed by detail coefficient tuple for each level
    
    """
    return pywt.wavedec2(img, wavelet, mode='symmetric', level=level, axes=(-2, -1))


def waverec(coeffs, wavelet):
    """Reconstruct an image using a multilevel 2D inverse discrete wavelet transform
    
    Parameters
    ----------
    coeffs : list
        the approximation coefficients followed by detail coefficient tuple for each level
    wavelet : str
        name of the mother wavelet

    Returns
    -------
    img : ndarray
        reconstructed image

    """
    return pywt.waverec2(coeffs, wavelet, mode='symmetric', axes=(-2, -1))


def fft(data, axis=-1, shift=True):
    """Computes the 1D Fast Fourier Transform of an input array
    
    Parameters
    ----------
    data : ndarray
        input array to transform
    axis : int (optional)
        axis to perform the 1D FFT over
    shift : bool
        indicator for centering the DC component

    Returns
    -------
    fdata : ndarray
        transformed data

    """
    fdata = fftpack.rfft(data, axis=axis)
    if shift:
        fdata = fftpack.fftshift(fdata)
    return fdata


def ifft(fdata, axis=-1):
    return fftpack.irfft(fdata, axis=axis)


def fft2(data, shift=True):
    """Computes the 2D Fast Fourier Transform of an input array
    
    Parameters
    ----------
    data : ndarray
        data to transform
    shift : bool
        indicator for center the DC component

    Returns
    -------
    fdata : ndarray
        transformed data

    """
    fdata = fftpack.fft2(data)
    if shift:
        fdata = fftpack.fftshift(fdata)
    return fdata


def notch(n, sigma):
    """Generates a 1D gaussian notch filter `n` pixels long
    
    Parameters
    ----------
    n : int
        length of the gaussian notch filter
    sigma : float
        notch width

    Returns
    -------
    g : ndarray
        (n,) array containing the gaussian notch filter

    """
    if n <= 0:
        raise ValueError('n must be positive')
    else:
        n = int(n)
    if sigma <= 0:
        raise ValueError('sigma must be positive')
    x = np.arange(n)
    g = 1 - np.exp(-x ** 2 / (2 * sigma ** 2))
    return g


def gaussian_filter(shape, sigma):
    """Create a gaussian notch filter
    
    Parameters
    ----------
    shape : tuple
        shape of the output filter
    sigma : float
        filter bandwidth

    Returns
    -------
    g : ndarray
        the impulse response of the gaussian notch filter

    """
    g = notch(n=shape[-1], sigma=sigma)
    return np.broadcast_to(g, shape)


def filter_streaks(img, sigma, level=0, wavelet='db2'):
    """Filter horizontal streaks using wavelet-FFT filter

    Parameters
    ----------
    img : ndarray
        input image array to filter
    sigma : float
        filter bandwidth (larger for more filtering)
    level : int
        number of wavelet levels to use
    wavelet : str
        name of the mother wavelet

    Returns
    -------
    fimg : ndarray
        filtered image

    """
    if level == 0:
        coeffs = wavedec(img, wavelet)
    else:
        coeffs = wavedec(img, wavelet, level)
    approx = coeffs[0]
    detail = coeffs[1:]

    coeffs_filt = [approx]
    for ch, cv, cd in detail:
        fch = fft(ch, shift=False)
        g = gaussian_filter(shape=fch.shape, sigma=sigma)
        fch_filt = fch * g
        ch_filt = ifft(fch_filt)
        coeffs_filt.append((ch_filt, cv, cd))

    fimg = waverec(coeffs_filt, wavelet).astype(img.dtype)
    return fimg


def read_filter_save(input_path, output_path, sigma, level=0, wavelet='db2', compression=1):
    img = imread(str(input_path))
    fimg = filter_streaks(img, sigma, level=level, wavelet=wavelet)
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True)
    imsave(str(output_path), fimg, compression=compression)


def _read_filter_save(input_dict):
    input_path = input_dict['input_path']
    output_path = input_dict['output_path']
    sigma = input_dict['sigma']
    level = input_dict['level']
    wavelet = input_dict['wavelet']
    compression = input_dict['compression']
    read_filter_save(input_path, output_path, sigma, level, wavelet, compression)


def _find_all_images(input_path):
    input_path = Path(input_path)
    assert input_path.is_dir()
    img_paths = []
    for p in input_path.iterdir():
        if p.is_file():
            if p.suffix in supported_extensions:
                img_paths.append(p)
        elif p.is_dir():
            img_paths.extend(_find_all_images(p))
    return img_paths


def batch_filter(input_path, output_path, workers, chunks, sigma, level=0, wavelet='db2', compression=1):
    img_paths = _find_all_images(input_path)

    args = []
    for p in img_paths:
        rel_path = p.relative_to(input_path)
        o = output_path.joinpath(rel_path)
        arg_dict = {
            'input_path': p,
            'output_path': o,
            'sigma': sigma,
            'level': level,
            'wavelet': wavelet,
            'compression': compression
        }
        args.append(arg_dict)

    with multiprocessing.Pool(workers) as pool:
        list(tqdm.tqdm(pool.imap(_read_filter_save, args, chunksize=chunks), total=len(args)))


def _parse_args():
    parser = argparse.ArgumentParser(description="Streak elimination using wavelet and FFT filtering")
    parser.add_argument("--input", "-i", help="Path to input image or path", type=str, required=True)
    parser.add_argument("--sigma", "-s", help="Bandwidth (larger for more filtering)", type=float, required=True)
    parser.add_argument("--level", "-l", help="Number of decomposition levels", type=int, default=0)
    parser.add_argument("--wavelet", "-w", help="Name of the mother wavelet", type=str, default='db2')
    parser.add_argument("--workers", "-n", help="Number of workers (for batch processing)", type=int, default=1)
    parser.add_argument("--chunks", help="Chunk size (for batch processing)", type=int, default=1)
    parser.add_argument("--compression", "-c", help="Compression level for written tiffs", type=int, default=1)
    args = parser.parse_args()
    return args


def main():
    args = _parse_args()
    input_path = Path(args.input)
    if input_path.is_file():  # single image
        if input_path.suffix not in supported_extensions:
            print('Input file was found, but is not supported. Exiting...')
            return
        output_path = Path(input_path.parent).joinpath(input_path.stem+'_destriped'+input_path.suffix)
        read_filter_save(input_path,
                         output_path,
                         sigma=args.sigma,
                         level=args.level,
                         wavelet=args.wavelet,
                         compression=args.compression)
    elif input_path.is_dir():  # batch processing
        output_path = Path(input_path.parent).joinpath(str(input_path)+'_destriped')
        batch_filter(input_path,
                     output_path,
                     workers=args.workers,
                     chunks=args.chunks,
                     sigma=args.sigma,
                     level=args.level,
                     wavelet=args.wavelet,
                     compression=args.compression)
    else:
        print('Cannot find input file or directory. Exiting...')


if __name__ == "__main__":
    main()
