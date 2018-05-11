# pystripe

[![Travis CI Status](https://travis-ci.org/chunglabmit/pystripe.svg?branch=master)](https://travis-ci.org/chunglabmit/phathom)

An image processing package for removing streaks from SPIM images

```python
import pystripe

# filter a single image
fimg = pystripe.filter_streaks(img, sigma=4.0, level=7, wavelet='db2')

# batch process images in a directory (and subdirectories)
fimg = pystripe.batch_filter(input_path, 
                             output_path,
                             workers=8, 
                             sigma=4.0, 
                             level=7, 
                             wavelet='db2')
```

A typical result looks like this:

![Image](./demo/result.png?raw=true)

## Installation

Installation can be done using `pip`, e.g.

```bash
> pip install https://github.com/chunglabmit/pystripe/archive/master.zip --process-dependency-links --allow-external tsv
```

If using the provided Windows batch scripts, install within a conda environment
```bash
> conda create -n pystripe pip
> activate pystripe
> pip install https://github.com/chunglabmit/pystripe/archive/master.zip --process-dependency-links --allow-external tsv
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

* **--input**: (str) path to a single image or a directory with images to filter
* **--output, -o**: (str, optional) path to a single image or a directory to write to.
Parent directories will be created as needed. Note that setting the output to the input
will overwrite the original image(s). Default is either to write the result from `my/img/input.tif` to
`my/img/input_destriped.tif` or results from `my/folder/` to `my/folder_destriped/`. 
* **--sigma**: (float) bandwidth of the stripe filter
* **--level**: (int, optional) number of wavelet decomposition levels. Default is the maximum
possible given the image shape
* **--wavelet**: (str, optional) name of the mother wavelet. Default is `'db2'`. 
See [PyWavelets](https://pywavelets.readthedocs.io/en/latest/) for more options.
* **--workers**: (int, optional) number of cpu workers to use in batch processing. Default is cpu_count()
* **--chunks**: (int, optional) number of images each worker processes at a time. Default is 1
* **--compression**: (int, optional) compression level (0-9) for writing tiffs. Default is 1

## Batch script

The `scripts/` directory contains a Windows batch script for running `pystripe` within the
current working directory. In order to use the script as is, pystripe must be installed within a conda environment
named "pystripe" (see Installation). Also, the script itself **cannot** be named `pystripe.bat`
because Windows will assume the script is calling itself rather than the `pystripe.exe` on the
`PATH`. Feel free to adjust the parameters or add any of the additional arguments mentioned above.

## Authors
Phathom is maintained by members of the [Kwanghun Chung Lab](http://www.chunglab.org/) at MIT.
