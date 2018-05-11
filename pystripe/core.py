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
    return Path(path).suffix


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


def hist_match(source, template):
    """Adjust the pixel values of a grayscale image such that its histogram matches that of a target image

    Parameters
    ----------
    source: ndarray
        Image to transform; the histogram is computed over the flattened array
    template: ndarray
        Template image; can have different dimensions to source
    Returns
    -------
    matched: ndarray
        The transformed output image
        
    """

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)


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

    fimg = waverec(coeffs_filt, wavelet)

    scaled_fimg = hist_match(fimg, img)
    np.clip(scaled_fimg, np.iinfo(img.dtype).min, np.iinfo(img.dtype).max, out=scaled_fimg)
    return scaled_fimg.astype(img.dtype)


def read_filter_save(input_path, output_path, sigma, level=0, wavelet='db2', compression=1):
    """Convenience wrapper around filter streaks. Takes in a path to an image rather than an image array.
    
    Parameters
    ----------
    input_path : Path
        path to the image to filter
    output_path : Path
        path to write the result
    sigma : float
        bandwidth of the stripe filter
    level : int
        number of wavelet levels to use
    wavelet : str
        name of the mother wavelet
    compression : int
        compression level for writing tiffs

    """
    img = imread(str(input_path))
    fimg = filter_streaks(img, sigma, level=level, wavelet=wavelet)
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True)
    imsave(str(output_path), fimg, compression=compression)


def _read_filter_save(input_dict):
    """Same as `read_filter_save' but with a single input dictionary. Used for pool.imap() in batch_filter
    
    Parameters
    ----------
    input_dict : dict
        input dictionary with arguments for `read_filter_save`.

    """
    input_path = input_dict['input_path']
    output_path = input_dict['output_path']
    sigma = input_dict['sigma']
    level = input_dict['level']
    wavelet = input_dict['wavelet']
    compression = input_dict['compression']
    read_filter_save(input_path, output_path, sigma, level, wavelet, compression)


def _find_all_images(input_path):
    """Find all images with a supported file extension within a directory and all its subdirectories.
    
    Parameters
    ----------
    input_path : str
        root directory to start image search

    Returns
    -------
    img_paths : list
        a list of Path objects for all found images

    """
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
    """Applies `streak_filter` to all images in `input_path` and write the results to `output_path`.
    
    Parameters
    ----------
    input_path : Path
        root directory to search for images to filter
    output_path : Path
        root directory for writing results
    workers : int
        number of CPU workers to use
    chunks : int
        number of images for each CPU to process at a time
    sigma : float
        bandwidth of the stripe filter
    level : int
        number of wavelet levels to use
    wavelet : str
        name of the mother wavelet
    compression : int
        compression level to use in tiff writing

    """
    if workers == 0:
        workers = multiprocessing.cpu_count()

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
    parser.add_argument("--output", "-o", help="Path to output image or path", type=str, default='')
    parser.add_argument("--sigma", "-s", help="Bandwidth (larger for more filtering)", type=float, required=True)
    parser.add_argument("--level", "-l", help="Number of decomposition levels", type=int, default=0)
    parser.add_argument("--wavelet", "-w", help="Name of the mother wavelet", type=str, default='db2')
    parser.add_argument("--workers", "-n", help="Number of workers (for batch processing)", type=int, default=0)
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
        if args.output == '':
            output_path = Path(input_path.parent).joinpath(input_path.stem+'_destriped'+input_path.suffix)
        else:
            output_path = Path(args.output)
            assert output_path.suffix in supported_extensions
        read_filter_save(input_path,
                         output_path,
                         sigma=args.sigma,
                         level=args.level,
                         wavelet=args.wavelet,
                         compression=args.compression)
    elif input_path.is_dir():  # batch processing
        if args.output == '':
            output_path = Path(input_path.parent).joinpath(str(input_path)+'_destriped')
        else:
            output_path = Path(args.output)
            assert output_path.suffix == ''
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
