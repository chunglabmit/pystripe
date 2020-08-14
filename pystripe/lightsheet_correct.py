"""
LightsheetCorrection
====================

Module to remove lightsheet artifacts in images.

Adapted from https://github.com/ChristophKirst/ClearMap2

Kirst, et al. "Mapping the Fine-Scale Organization and Plasticity of the Brain Vasculature." 
Cell 180.4 (2020): 780-795.
https://doi.org/10.1016/j.cell.2020.01.028

Renier et al. "Mapping of brain activity by automated volume analysis of immediate early genes."
Cell 165.7 (2016): 1789-1802.
https://doi.org/10.1016/j.cell.2016.05.007
"""


import numpy as np
import scipy.ndimage as ndi


###############################################################################
### Lightsheet correction
###############################################################################

def correct_lightsheet(source, percentile=0.25, max_bin=2 ** 12, mask=None,
                       lightsheet=dict(selem=(150, 1, 1)),
                       background=dict(selem=(200, 200, 1),
                       spacing=(25, 25, 1),
                       interpolate=1,
                       dtype=float,
                       step=(2, 2, 1)),
                       lightsheet_vs_background=2,
                       return_lightsheet=False, return_background=False):
    """Removes lightsheet artifacts.

    Arguments
    ---------
    source : array
      The source to correct.
    percentile : float in [0,1]
      Ther percentile to base the lightsheet correction on.
    mask : array or None
      Optional mask.
    lightsheet : dict
      Parameter to pass to the percentile routine for the lightsheet artifact
      estimate. See :func:`ImageProcessing.Filter.Rank.percentile`.
    background : dict
      Parameter to pass to the percentile rouitne for the background estimation.
    lightsheet_vs_background : float
      The background is multiplied by this weight before comparing to the
      lightsheet artifact estimate.
    return_lightsheet : bool
      If True, return the lightsheeet artifact estimate.
    return_background : bool
      If True, return the background estimate.

    Returns
    -------
    corrected : array
      Lightsheet artifact corrected image.
    lightsheet : array
      The lightsheet artifact estimate.
    background : array
      The background estimate.

    Note
    ----
    The routine implements a fast but efftice way to remove lightsheet artifacts.
    Effectively the percentile in an eleoganted structural element along the
    lightsheet direction centered around each pixel is calculated and then
    compared to the percentile in a symmetrical box like structural element
    at the same pixel. The former is an estimate of the lightsheet artifact
    the latter of the backgrond. The background is multiplied by the factor
    lightsheet_vs_background and then the minimum of both results is subtracted
    from the source.
    Adding an overall background estimate helps to not accidentally remove
    vessesl like structures along the light-sheet direction.
    """

    # lightsheet artifact estimate
    l = local_percentile(source, percentile=percentile, mask=mask, **lightsheet)
    # background estimate
    b = local_percentile(source, percentile=percentile, mask=mask, **background);
    # combined estimate
    lb = np.minimum(l, lightsheet_vs_background * b);
    # corrected image
    c = source - np.minimum(source, lb);

    result = (c,)
    if return_lightsheet:
        result += (l,)
    if return_background:
        result += (b,)
    if len(result) == 1:
        result = result[0]
    return result


###############################################################################
### Local image processing for Local Statistics
###############################################################################

def apply_local_function(source, function, selem=(50, 50), spacing=None, step=None, interpolate=2, mask=None,
                         fshape=None, dtype=None, return_centers=False):
    """Calculate local histograms on a sub-grid, apply a scalar valued function and resmaple to original image shape.

    Arguments
    ---------
    source : array
      The source to process.
    function : function
      Function to apply to the linear array of the local source data.
      If the function does not return a scalar, fshape has to be given.
    selem : tuple or array or None
      The structural element to use to extract the local image data.
      If tuple, use a rectangular region of this shape. If array, the array
      is assumed to be bool and acts as a local mask around the center point.
    spacing : tuple or None
      The spacing between sample points. If None, use shape of selem.
    step : tuple of int or None
      If tuple, subsample the local region by these step. Note that the
      selem is applied after this subsampling.
    interpolate : int or None
      If int, resample the result back to the original source shape using this
      order of interpolation. If None, return the results on the sub-grid.
    mask : array or None
      Optional mask to use.
    fshape : tuple or None
      If tuple, this is the shape of the function output.
      If None assumed to be (1,).
    dtype : dtype or None
      Optional data type for the result.
    return_centers : bool
      If True, additionaly return the centers of the sampling.

    Returns
    -------
    local : array
      The reuslt of applying the function to the local samples.
    cetners : array
      Optional cttners of the sampling.
    """

    if spacing is None:
        spacing = selem;
    shape = source.shape;
    ndim = len(shape);

    if step is None:
        step = (None,) * ndim

    if len(spacing) != ndim or len(step) != ndim:
        raise ValueError('Dimension mismatch in the parameters!')

        # histogram centers
    n_centers = tuple(s // h for s, h in zip(shape, spacing))
    left = tuple((s - (n - 1) * h) // 2 for s, n, h in zip(shape, n_centers, spacing));

    # center points
    centers = np.array(np.meshgrid(*[range(l, s, h) for l, s, h in zip(left, shape, spacing)], indexing='ij'));
    # centers = np.reshape(np.moveaxis(centers, 0, -1),(-1,len(shape)));
    centers = np.moveaxis(centers, 0, -1)

    # create result
    rshape = (1,) if fshape is None else fshape;
    rdtype = source.dtype if dtype is None else dtype;
    results = np.zeros(n_centers + rshape, dtype=rdtype);

    # calculate function
    centers_flat = np.reshape(centers, (-1, ndim));
    results_flat = np.reshape(results, (-1,) + rshape);

    # structuring element
    if isinstance(selem, np.ndarray):
        selem_shape = selem.shape
    else:
        selem_shape = selem;
        selem = None

    hshape_left = tuple(h // 2 for h in selem_shape);
    hshape_right = tuple(h - l for h, l in zip(selem_shape, hshape_left));

    for result, center in zip(results_flat, centers_flat):
        sl = tuple(slice(max(0, c - l), min(c + r, s), d) for c, l, r, s, d in
                   zip(center, hshape_left, hshape_right, shape, step));
        if selem is None:
            if mask is not None:
                data = source[sl][mask[sl]];
            else:
                data = source[sl].flatten();
        else:
            slm = tuple(
                slice(None if c - l >= 0 else min(l - c, m), None if c + r <= s else min(m - (c + r - s), m), d) for
                c, l, r, s, d, m in zip(center, hshape_left, hshape_right, shape, step, selem_shape));
            data = source[sl];
            if mask is not None:
                data = data[np.logical_and(mask[sl], selem[slm])]
            else:
                data = data[selem[slm]];

        # print result.shape, data.shape, function(data)
        result[:] = function(data);

    # resample
    if interpolate:
        res_shape = results.shape[:len(shape)];
        zoom = tuple(float(s) / float(r) for s, r in zip(shape, res_shape));
        results_flat = np.reshape(results, res_shape + (-1,));
        results_flat = np.moveaxis(results_flat, -1, 0);
        full = np.zeros(shape + rshape, dtype=results.dtype);
        full_flat = np.reshape(full, shape + (-1,));
        full_flat = np.moveaxis(full_flat, -1, 0);
        # print results_flat.shape, full_flat.shape
        for r, f in zip(results_flat, full_flat):
            f[:] = ndi.zoom(r, zoom=zoom, order=interpolate);
        results = full;

    if fshape is None:
        results.shape = results.shape[:-1];

    if return_centers:
        return results, centers
    else:
        return results


def local_percentile(source, percentile, selem=(50, 50), spacing=None, step=None, interpolate=1, mask=None, dtype=None,
                     return_centers=False):
    """Calculate local percentile.

    Arguments
    ---------
    source : array
      The source to process.
    percentile : float or array
      The percentile(s) to estimate locally.
    selem : tuple or array or None
      The structural element to use to extract the local image data.
      If tuple, use a rectangular region of this shape. If array, the array
      is assumed to be bool and acts as a local mask around the center point.
    spacing : tuple or None
      The spacing between sample points. If None, use shape of selem.
    step : tuple of int or None
      If tuple, subsample the local region by these step. Note that the
      selem is applied after this subsampling.
    interpolate : int or None
      If int, resample the result back to the original source shape using this
      order of interpolation. If None, return the results on the sub-grid.
    mask : array or None
      Optional mask to use.
    return_centers : bool
      If True, additionaly return the centers of the sampling.

    Returns
    -------
    percentiles : array
      The local percentiles.
    cetners : array
      Optional centers of the sampling.
    """
    if isinstance(percentile, (tuple, list)):
        percentile = np.array([100 * p for p in percentile]);
        fshape = (len(percentile),)

        def _percentile(data):
            if len(data) == 0:
                return np.array((0,) * len(percentile));
            return np.percentile(data, percentile, axis=None);

    else:
        percentile = 100 * percentile;
        fshape = None;

        def _percentile(data):
            if len(data) == 0:
                return 0;
            return np.percentile(data, percentile, axis=None);

    return apply_local_function(source, selem=selem, spacing=spacing, step=step, interpolate=interpolate, mask=mask,
                                dtype=dtype, return_centers=return_centers,
                                function=_percentile, fshape=fshape);


