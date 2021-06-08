import argparse
from argparse import RawDescriptionHelpFormatter
from pathlib import Path
import os
import numpy as np
from scipy import fftpack, ndimage
from skimage.filters import threshold_otsu
import tifffile
import pywt
import multiprocessing
import tqdm
from dcimg import DCIMGFile
from pystripe import raw
from .lightsheet_correct import correct_lightsheet
import warnings
warnings.filterwarnings("ignore")


supported_extensions = ['.tif', '.tiff', '.raw', '.dcimg']
nb_retry = 10


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


def imread_dcimg(path, z):
    """Load a slice from a DCIMG file

    Parameters
    ------------
    path : str
        path to DCIMG file
    z : int
        z slice index to load

    Returns
    --------
    img : ndarray
        image as numpy array

    """
    with DCIMGFile(path) as arr:
        img = arr[z]
    return img


def check_dcimg_shape(path):
    """Returns the image shape of a DCIMG file

    Parameters
    ------------
    path : str
        path to DCIMG file

    Returns
    --------
    shape : tuple
        image shape

    """
    with DCIMGFile(path) as arr:
        shape = arr.shape
    return shape


def check_dcimg_start(path):
    """Returns the starting z position of a DCIMG substack.

    This function assumes a zero-padded 6 digit filename in tenths of micron.
    For example, `0015250.dicom` would indicate a substack starting at z = 1525 um.

    Parameters
    ------------
    path : str
        path to DCIMG file

    Returns
    --------
    start : int
        starting z position in tenths of micron

    """
    return int(os.path.basename(path).split('.')[0])


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
    # fdata = fftpack.rfft(fdata, axis=0)
    if shift:
        fdata = fftpack.fftshift(fdata)
    return fdata


def ifft(fdata, axis=-1):
    # fdata = fftpack.irfft(fdata, axis=0)
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


def ifft2(fdata):
    return fftpack.ifft2(fdata)


def magnitude(fdata):
    return np.sqrt(np.real(fdata) ** 2 + np.imag(fdata) ** 2)


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
    g_mask = np.broadcast_to(g, shape).copy()
    return g_mask


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


def max_level(min_len, wavelet):
    w = pywt.Wavelet(wavelet)
    return pywt.dwt_max_level(min_len, w.dec_len)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# def sigmoid(x):
#     if x >= 0:
#         z = np.exp(-x)
#         return 1 / (1 + z)
#     else:
#         z = np.exp(x)
#         return z / (1 + z)


def foreground_fraction(img, center, crossover, smoothing):
    z = (img-center)/crossover
    f = sigmoid(z)
    return ndimage.gaussian_filter(f, sigma=smoothing)


def filter_subband(img, sigma, level, wavelet):
    img_log = np.log(1 + img)

    if level == 0:
        coeffs = wavedec(img_log, wavelet)
    else:
        coeffs = wavedec(img_log, wavelet, level)
    approx = coeffs[0]
    detail = coeffs[1:]

    width_frac = sigma / img.shape[0]
    coeffs_filt = [approx]
    for ch, cv, cd in detail:
        s = ch.shape[0] * width_frac
        fch = fft(ch, shift=False)
        g = gaussian_filter(shape=fch.shape, sigma=s)
        fch_filt = fch * g
        ch_filt = ifft(fch_filt)
        coeffs_filt.append((ch_filt, cv, cd))

    img_log_filtered = waverec(coeffs_filt, wavelet)
    return np.exp(img_log_filtered)-1


def apply_flat(img, flat):
    return (img / flat).astype(img.dtype)


def filter_streaks(img, sigma, level=0, wavelet='db3', crossover=10, threshold=-1, flat=None, dark=0):
    """Filter horizontal streaks using wavelet-FFT filter

    Parameters
    ----------
    img : ndarray
        input image array to filter
    sigma : float or list
        filter bandwidth(s) in pixels (larger gives more filtering)
    level : int
        number of wavelet levels to use
    wavelet : str
        name of the mother wavelet
    crossover : float
        intensity range to switch between filtered background and unfiltered foreground
    threshold : float
        intensity value to separate background from foreground. Default is Otsu
    flat : ndarray
        reference image for illumination correction. Must be same shape as input images. Default is None
    dark : float
        Intensity to subtract from the images for dark offset. Default is 0.

    Returns
    -------
    fimg : ndarray
        filtered image

    """
    smoothing = 1

    if threshold == -1:
        try:
            threshold = threshold_otsu(img)
        except ValueError:
            threshold = 1

    img = np.array(img, dtype=np.float)
    #
    # Need to pad image to multiple of 2
    #
    pady, padx = [_ % 2 for _ in img.shape]
    if pady == 1 or padx == 1:
        img = np.pad(img, ((0, pady), (0, padx)), mode="edge")

    # TODO: Clean up this logic with some dual-band CLI alternative
    sigma1 = sigma[0]  # foreground
    sigma2 = sigma[1]  # background
    if sigma1 > 0:
        if sigma2 > 0:
            if sigma1 == sigma2:  # Single band
                fimg = filter_subband(img, sigma1, level, wavelet)
            else:  # Dual-band
                background = np.clip(img, None, threshold)
                foreground = np.clip(img, threshold, None)
                background_filtered = filter_subband(background, sigma[1], level, wavelet)
                foreground_filtered = filter_subband(foreground, sigma[0], level, wavelet)
                # Smoothed homotopy
                f = foreground_fraction(img, threshold, crossover, smoothing=1)
                fimg = foreground_filtered * f + background_filtered * (1 - f)
        else:  # Foreground filter only
            foreground = np.clip(img, threshold, None)
            foreground_filtered = filter_subband(foreground, sigma[0], level, wavelet)
            # Smoothed homotopy
            f = foreground_fraction(img, threshold, crossover, smoothing=1)
            fimg = foreground_filtered * f + img * (1 - f)
    else:
        if sigma2 > 0:  # Background filter only
            background = np.clip(img, None, threshold)
            background_filtered = filter_subband(background, sigma[1], level, wavelet)
            # Smoothed homotopy
            f = foreground_fraction(img, threshold, crossover, smoothing=1)
            fimg = img * f + background_filtered * (1 - f)
        else:
            # sigma1 and sigma2 are both 0, so skip the destriping
            fimg = img

    # TODO: Fix code to clip back to original bit depth
    # scaled_fimg = hist_match(fimg, img)
    # np.clip(scaled_fimg, np.iinfo(img.dtype).min, np.iinfo(img.dtype).max, out=scaled_fimg)

    # Subtract the dark offset fiirst
    if dark > 0:
        fimg = fimg - dark

    # Divide by the flat
    if flat is not None:
        fimg = apply_flat(fimg, flat)

    # Convert to 16 bit image
    np.clip(fimg, 0, 2**16 - 1, out=fimg)  # Clip to 16-bit unsigned range
    fimg = fimg.astype('uint16')

    if padx > 0:
        fimg = fimg[:, :-padx]
    if pady > 0:
        fimg = fimg[:-pady]
    return fimg


def read_filter_save(input_path, output_path, sigma, level=0, wavelet='db3',
                     crossover=10, threshold=-1, compression=1,
                     flat=None, dark=0, z_idx=None, rotate=False,
                     lightsheet=False,
                     artifact_length=150,
                     background_window_size=200,
                     percentile=.25,
                     lightsheet_vs_background=2.0,
                     dont_convert_16bit=False):

    """Convenience wrapper around filter streaks. Takes in a path to an image rather than an image array

    Note that the directory being written to must already exist before calling this function

    Parameters
    ----------
    input_path : Path
        path to the image to filter
    output_path : Path
        path to write the result
    sigma : list
        bandwidth of the stripe filter
    level : int
        number of wavelet levels to use
    wavelet : str
        name of the mother wavelet
    crossover : float
        intensity range to switch between filtered background and unfiltered foreground
    threshold : float
        intensity value to separate background from foreground. Default is Otsu
    compression : int
        compression level for writing tiffs
    flat : ndarray
        reference image for illumination correction. Must be same shape as input images. Default is None
    dark : float
        Intensity to subtract from the images for dark offset. Default is 0.
    z_idx : int
        z index of DCIMG slice. Only applicable to DCIMG files.
    rotate : bool
        rotate x and y if true
    lightsheet : bool
        if False, use wavelet method, if true use correct_lightsheet
    artifact_length : int
        # of pixels to look at in the lightsheet direction
    background_window_size : int
        Look at this size window around the pixel in x and y
    percentile : float
        Take this percentile as background with lightsheet
    lightsheet_vs_background : float
        weighting factor to use background or lightsheet background
    dont_convert_16bit : bool
        Flag for converting to 16-bit
    """
    if z_idx is None:
        # Path must be TIFF or RAW
        img = imread(str(input_path))
        dtype = img.dtype
        if not dont_convert_16bit:
            dtype = np.uint16
    else:
        # Path must be to DCIMG file
        assert str(input_path).endswith('.dcimg')
        img = imread_dcimg(str(input_path), z_idx)
        dtype = np.uint16
    if rotate:
        img = np.rot90(img)
    if not lightsheet:
        fimg = filter_streaks(img, sigma, level=level, wavelet=wavelet, crossover=crossover, threshold=threshold, flat=flat, dark=dark)
    else:
        fimg = correct_lightsheet(
            img.reshape(img.shape[0], img.shape[1], 1),
            percentile=percentile,
            lightsheet=dict(selem=(1, artifact_length, 1)),
            background=dict(
                selem=(background_window_size, background_window_size, 1),
                spacing=(25, 25, 1),
                interpolate=1,
                dtype=np.float32,
                step=(2, 2, 1)),
            lightsheet_vs_background=lightsheet_vs_background
            ).reshape(img.shape[0], img.shape[1])
        if flat is not None:
            fimg = apply_flat(fimg, flat)
    # Save image, retry if OSError for NAS
    for _ in range(nb_retry):
        try:
            imsave(str(output_path), fimg.astype(dtype), compression=compression)
        except OSError:
            print('Retrying...')
            continue
        break


def _read_filter_save(input_dict):
    """Same as `read_filter_save' but with a single input dictionary. Used for pool.imap() in batch_filter

    Parameters
    ----------
    input_dict : dict
        input dictionary with arguments for `read_filter_save`.

    """
    # input_path = input_dict['input_path']
    # output_path = input_dict['output_path']
    # sigma = input_dict['sigma']
    # level = input_dict['level']
    # wavelet = input_dict['wavelet']
    # crossover = input_dict['crossover']
    # threshold = input_dict['threshold']
    # compression = input_dict['compression']
    # flat = input_dict['flat']
    # read_filter_save(input_path, output_path, sigma, level, wavelet, crossover, threshold, compression, flat)
    read_filter_save(**input_dict)


def _find_all_images(input_path, zstep=None):
    """Find all images with a supported file extension within a directory and all its subdirectories

    Parameters
    ----------
    input_path : path-like
        root directory to start image search
    zstep : int
        step-size for DCIMG stacks in tenths of micron

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
                if p.suffix == '.dcimg':
                    if zstep is None:
                        raise ValueError('Unknown zstep for DCIMG slice positions')
                    shape = check_dcimg_shape(str(p))
                    start = check_dcimg_start(str(p))
                    substack = [(p, i, start + i * zstep) for i in range(shape[0])]
                    img_paths += substack
                else:
                    img_paths.append(p)
        elif p.is_dir():
            img_paths.extend(_find_all_images(p, zstep))
    return img_paths


def batch_filter(input_path, output_path, workers, chunks, sigma, level=0, wavelet='db3', crossover=10,
                 threshold=-1, compression=1, flat=None, dark=0, zstep=None, rotate=False,
                 lightsheet=False,
                 artifact_length=150,
                 background_window_size=200,
                 percentile=.25,
                 lightsheet_vs_background=2.0,
                 dont_convert_16bit=False
                 ):
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
    sigma : list
        bandwidth of the stripe filter in pixels
    level : int
        number of wavelet levels to use
    wavelet : str
        name of the mother wavelet
    crossover : float
        intensity range to switch between filtered background and unfiltered foreground. Default: 100 a.u.
    threshold : float
        intensity value to separate background from foreground. Default is Otsu
    compression : int
        compression level to use in tiff writing
    flat : ndarray
        reference image for illumination correction. Must be same shape as input images. Default is None
    dark : float
        Intensity to subtract from the images for dark offset. Default is 0.
    zstep : int
        Zstep in tenths of micron. only used for DCIMG files.
    rotate : bool
        Flag for 90 degree rotation.
    dont_convert_16bit : bool
        Flag for converting to 16-bit
    """
    if workers == 0:
        workers = multiprocessing.cpu_count()
    print('Looking for images in {}...'.format(input_path))
    img_paths = _find_all_images(input_path, zstep)
    print('Found {} compatible images'.format(len(img_paths)))
    print('Setting up {} workers...'.format(workers))
    args = []
    for p in img_paths:
        if isinstance(p, tuple):  # DCIMG found
            p, z_idx, z = p
            rel_path = p.relative_to(input_path).parent.joinpath('{:04d}.tif'.format(z))
        else:  # TIFF or RAW found
            z_idx = None
            rel_path = p.relative_to(input_path)
        o = output_path.joinpath(rel_path)
        if not o.parent.exists():
            o.parent.mkdir(parents=True)
        arg_dict = {
            'input_path': p,
            'output_path': o,
            'sigma': sigma,
            'level': level,
            'wavelet': wavelet,
            'crossover': crossover,
            'threshold': threshold,
            'compression': compression,
            'flat': flat,
            'dark': dark,
            'z_idx': z_idx,
            'rotate': rotate,
            'lightsheet': lightsheet,
            'artifact_length': artifact_length,
            'background_window_size': background_window_size,
            'percentile': percentile,
            'lightsheet_vs_background': lightsheet_vs_background,
            'dont_convert_16bit' : dont_convert_16bit
        }
        args.append(arg_dict)
    print('Pystripe batch processing progress:')
    with multiprocessing.Pool(workers) as pool:
        list(tqdm.tqdm(pool.imap(_read_filter_save, args, chunksize=chunks), total=len(args), ascii=True))
    print('Done!')


def normalize_flat(flat):
    flat_float = flat.astype(np.float32)
    return flat_float / flat_float.max()


def _parse_args():
    parser = argparse.ArgumentParser(description="Pystripe (version 0.2.0)\n\n"
        "If only sigma1 is specified, only foreground of the images will be filtered.\n"
        "If sigma2 is specified and sigma1 = 0, only the background of the images will be filtered.\n"
        "If sigma1 == sigma2 > 0, input images will not be split before filtering.\n"
        "If sigma1 != sigma2, foreground and backgrounds will be filtered separately.\n"
        "The crossover parameter defines the width of the transistion between the filtered foreground and background",
                                     formatter_class=RawDescriptionHelpFormatter,
                                     epilog='Developed 2018 by Justin Swaney, Kwanghun Chung Lab\n'
                                            'Massachusetts Institute of Technology\n')
    parser.add_argument("--input", "-i", help="Path to input image or path", type=str, required=True)
    parser.add_argument("--output", "-o", help="Path to output image or path (Default: x_destriped)", type=str, default='')
    parser.add_argument("--sigma1", "-s1", help="Foreground bandwidth [pixels], larger = more filtering", type=float, default=0)
    parser.add_argument("--sigma2", "-s2", help="Background bandwidth [pixels] (Default: 0, off)", type=float, default=0)
    parser.add_argument("--level", "-l", help="Number of decomposition levels (Default: max possible)", type=int, default=0)
    parser.add_argument("--wavelet", "-w", help="Name of the mother wavelet (Default: Daubechies 3 tap)", type=str, default='db3')
    parser.add_argument("--threshold", "-t", help="Global threshold value (Default: -1, Otsu)", type=float, default=-1)
    parser.add_argument("--crossover", "-x", help="Intensity range to switch between foreground and background (Default: 10)", type=float, default=10)
    parser.add_argument("--workers", "-n", help="Number of workers for batch processing (Default: # CPU cores)", type=int, default=0)
    parser.add_argument("--chunks", help="Chunk size for batch processing (Default: 1)", type=int, default=1)
    parser.add_argument("--compression", "-c", help="Compression level for written tiffs (Default: 1)", type=int, default=1)
    parser.add_argument("--flat", "-f", help="Flat reference TIFF image of illumination pattern used for correction", type=str, default=None)
    parser.add_argument("--dark", "-d", help="Intensity of dark offset in flat-field correction", type=float, default=0)
    parser.add_argument("--zstep", "-z", help="Z-step in micron. Only used for DCIMG files.", type=float, default=None)
    parser.add_argument("--rotate", "-r", help="Rotate output images 90 degrees counter-clockwise", action='store_true')
    parser.add_argument("--lightsheet", help="Use the lightsheet method", action="store_true")
    parser.add_argument("--artifact-length", help="Look for minimum in lightsheet direction over this length", default=150, type=int)
    parser.add_argument("--background-window-size", help="Size of window in x and y for background estimation", default=200, type=int)
    parser.add_argument("--percentile", help="The percentile at which to measure the background", type=float, default=.25)
    parser.add_argument("--lightsheet-vs-background", help="The background is multiplied by this weight when comparing lightsheet against background", type=float, default=2.0)
    parser.add_argument("--dont-convert-16bit", help="Is the output converted to 16-bit .tiff or not", action="store_true")
    args = parser.parse_args()
    return args


def main():
    args = _parse_args()
    sigma = [args.sigma1, args.sigma2]
    input_path = Path(args.input)

    flat = None
    if args.flat is not None:
        flat = normalize_flat(imread(args.flat))

    zstep = None
    if args.zstep is not None:
        zstep = int(args.zstep * 10)

    if args.dark < 0:
        raise ValueError('Only positive values for dark offset are allowed')

    if input_path.is_file():  # single image
        if input_path.suffix not in supported_extensions:
            print('Input file was found but is not supported. Exiting...')
            return
        if args.output == '':
            output_path = Path(input_path.parent).joinpath(input_path.stem+'_destriped'+input_path.suffix)
        else:
            output_path = Path(args.output)
            assert output_path.suffix in supported_extensions
        read_filter_save(input_path,
                         output_path,
                         sigma=sigma,
                         level=args.level,
                         wavelet=args.wavelet,
                         crossover=args.crossover,
                         threshold=args.threshold,
                         compression=args.compression,
                         flat=flat,
                         dark=args.dark,
                         rotate=args.rotate,  # Does not work on DCIMG files
                         lightsheet=args.lightsheet,
                         artifact_length=args.artifact_length,
                         background_window_size=args.background_window_size,
                         percentile=args.percentile,
                         lightsheet_vs_background=args.lightsheet_vs_background,
                         dont_convert_16bit=args.dont_convert_16bit
                         )
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
                     sigma=sigma,
                     level=args.level,
                     wavelet=args.wavelet,
                     crossover=args.crossover,
                     threshold=args.threshold,
                     compression=args.compression,
                     flat=flat,
                     dark=args.dark,
                     zstep=zstep,
                     rotate=args.rotate,
                     lightsheet=args.lightsheet,
                     artifact_length=args.artifact_length,
                     background_window_size=args.background_window_size,
                     percentile=args.percentile,
                     lightsheet_vs_background=args.lightsheet_vs_background,
                     dont_convert_16bit=args.dont_convert_16bit
                     )
    else:
        print('Cannot find input file or directory. Exiting...')


if __name__ == "__main__":
    main()
