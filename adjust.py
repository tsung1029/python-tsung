"""
This file collects functions that adjust the original data
Adjustments include resizing, transposing, rearanging the data array and change
the axis data accordingly.

Note that in this file we do not modify existing data stored in the array other
than shifting them aroud or neglecting part of the data.
"""

import numpy as np
import str2keywords
from h5_utilities import hdf_data
from arraymask import mask
from scipy.signal import hilbert


def subrange(data_in, npts_start=None, npts_end=None, axes=None):
    """
    obtain a subarray of the data array

    :param data_in: array_like
    :param npts_start: int or ints marks starting index
    :param npts_end: int or ints marks ending index
    :param axes: change the axes accordingly
    :return: data and axes (if provided)
    """
    if isinstance(data_in, hdf_data):
        data = data_in.data
        axes = data_in.axes
    else:
        data = data_in
    if npts_start or npts_end:
        ndim = data.ndim
        if npts_start:
            if type(npts_start) == int:
                npts_start = ((npts_start,) * ndim)
        else:
            npts_start = ((0,) * ndim)
        if npts_end:
            if type(npts_end) == int:
                npts_end = ((npts_end,) * ndim)
        else:
            npts_end = data.shape
        # need to update the metadata of hdf later
        if isinstance(data_in, hdf_data):
            xdim = np.size(data_in.XMAX)
            ind = np.arange(xdim)
            pos = 0
            for i, xmax in enumerate(data_in.XMAX):
                for j in xrange(pos, ndim):
                    if xmax == axes[j].axis_max:
                        pos = j + 1
                        ind[i] = j
        # chcek if the range is legit
        if npts_start < npts_end < data.shape:
            if ndim == 2:
                data = np.copy(data[npts_start[0]:npts_end[0], npts_start[1]:npts_end[1]])
            elif ndim == 3:
                data = data[npts_start[0]:npts_end[0], npts_start[1]:npts_end[1], npts_start[2]:npts_end[2]]
            elif ndim == 1:
                data = data[npts_start[0]:npts_end[0]]
            elif ndim == 4:
                data = data[npts_start[0]:npts_end[0], npts_start[1]:npts_end[1],
                            npts_start[2]:npts_end[2], npts_start[3]:npts_end[3]]
        else:
            print "illegal subrange, do nothing."
        if axes:
            for di in xrange(1, ndim+1):
                axes[-di].axis_min += axes[-di].increment * npts_start[di-1]
                axes[-di].axis_numberpoints = npts_end[di-1] - npts_start[di-1]
                axes[-di].axis_max = axes[-di].increment * axes[-di].axis_numberpoints + axes[-di].axis_min
    if isinstance(data_in, hdf_data):
        data_in.data = data
        data_in.axes = axes
        data_in.shape = data.shape
        if xdim == 2:
            data_in.XMAX = np.array([axes[ind[0]].axis_max, axes[ind[1]].axis_max])
        elif xdim == 1:
            data_in.XMAX = np.array([axes[ind[0]].axis_max])
        elif xdim == 3:
            data_in.XMAX = np.array([axes[ind[0]].axis_max, axes[ind[1]].axis_max, axes[ind[2]].axis_max])
        return data_in
    if axes:
        return data, axes
    else:
        return data
