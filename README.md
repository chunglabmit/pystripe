# pystripe

[![Travis CI Status](https://travis-ci.org/chunglabmit/pystripe.svg?branch=master)](https://travis-ci.org/chunglabmit/pystripe)

An image processing package for removing streaks from SPIM images

Pystripe implements two different destriping algorithms. The first uses
wavelets to deconvolve along the striping direction. The second uses a
combination of background estimation from a linear patch in the striping
direction and a square background patch. It is adapted
from https://github.com/ChristophKirst/ClearMap2

Kirst, et al. "Mapping the Fine-Scale Organization and Plasticity of the Brain Vasculature." 
Cell 180.4 (2020): 780-795.
https://doi.org/10.1016/j.cell.2020.01.028

Renier et al. "Mapping of brain activity by automated volume analysis of immediate early genes."
Cell 165.7 (2016): 1789-1802.
https://doi.org/10.1016/j.cell.2016.05.007


```python
import pystripe

# filter a single image
fimg = pystripe.filter_streaks(img, sigma=[128, 256], level=7, wavelet='db2')

# batch process images in a directory (and subdirectories)
pystripe.batch_filter(input_path, 
                      output_path,
                      workers=8, 
                      sigma=[128, 256],  # foreground, background 
                      level=7, 
                      wavelet='db2')
```

A typical result looks like this:

![Image](./demo/result.jpg?raw=true)

## Installation

Installation can be done using `pip`, e.g.

```bash
> pip install https://github.com/chunglabmit/pystripe/archive/master.zip
```

If using the provided Windows batch scripts, install within a conda environment
```bash
> conda create -n pystripe python==3.6
> activate pystripe
> pip install https://github.com/chunglabmit/pystripe/archive/master.zip
```

## Command-line interface (CLI)

The following application is available from the command-line
after installing:

**pystripe**: batch streak elimination using wavelet and FFT filtering

This application filters horizontal streaks in input images using FFT filtering
of wavelet coefficients. When provided a single image path, it will filter the
provided image. When provided a directory, it will traverse the input directory
and filter all `.raw` or `.tif*` images. The resulting images will be saved
as (compressed) tiffs in a new folder next to the input directory with `_destriped`
appended.

Arguments for `pystripe` CLI:

* **--input, -i**: (str) path to a single image or a directory with images to filter
* **--output, -o**: (str, optional) path to a single image or a directory to write to.
Parent directories will be created as needed. Note that setting the output to the input
will overwrite the original image(s). Default is either to write the result from `my/img/input.tif` to
`my/img/input_destriped.tif` or results from `my/folder/` to `my/folder_destriped/`. 
* **--sigma1, -s1**: (float) bandwidth of the stripe filter for the foreground
* **--sigma2, -s2**: (float, optional) bandwidth of the stripe filter for the background. 
Default is 0, indicating no background destriping. 
If `sigma1 == sigma2 != 0`, then the image will not be decomposed into foreground and background images.
* **--level, -l**: (int, optional) number of wavelet decomposition levels. Default is the maximum
possible given the image shape
* **--wavelet, -w**: (str, optional) name of the mother wavelet. Default is `'db2'`. 
See [PyWavelets](https://pywavelets.readthedocs.io/en/latest/) for more options.
* **--crossover, -x** (float, optional) intensity range to transition between foreground and background bands.
Default is 10.
* **--workers, -n**: (int, optional) number of cpu workers to use in batch processing. Default is cpu_count()
* **--chunks**: (int, optional) number of images each worker processes at a time. Default is 1
* **--compression, -c**: (int, optional) compression level (0-9) for writing tiffs. Default is 1
* **--lightsheet**: (switch) if present, use the method of Kirst, et. al.
If absent, use wavelets
* **--artifact-length**: (int, optional) the length of the lightsheet line that
is used to estimate the background in the presence of lightsheet streaks
* **--background-window-size**: (int, optional) the size in x and y of the
background window to use as an alternate background estimation
* **--percentile**: (float, optional) the percentile at which to measure
background
* **--lightsheet-vs-background**: (float, optional) the weighting factor
to use when comparing the lightsheet and background estimates. Higher
favors the background method.

## Batch script

The `scripts/` directory contains a Windows batch script for running `pystripe` within the
current working directory. In order to use the script as is, pystripe must be installed within a conda environment
named "pystripe" (see Installation). Also, the script itself **cannot** be named `pystripe.bat` because Windows will assume the script is calling itself rather than the `pystripe.exe` on the
`PATH`. Feel free to adjust the parameters or add any of the additional arguments mentioned above.

## Authors
Pystripe is maintained by members of the [Kwanghun Chung Lab](http://www.chunglab.org/) at MIT.
