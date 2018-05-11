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
* **--sigma**: (float) bandwidth of the stripe filter
* **--level**: (int) number of wavelet decomposition levels
* **--wavelet**: (str) name of the mother wavelet
* **--workers**: (int) number of cpu workers to use in batch processing. Default is cpu_count()
* **--chunks**: (int) number of images each worker processes at a time
* **--compression**: (int) compression level (0-9) for writing tiffs

## Batch script

The `scripts/` directory contains batch scripts for running `pystripe` within the
current working directory. The parameters in these scripts can be adjusted as needed.

In order to use these scripts, pystripe must be installed within a conda environment
named "pystripe".

## Authors
Phathom is maintained by members of the [Kwanghun Chung Lab](http://www.chunglab.org/) at MIT.
