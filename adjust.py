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


def subrange_phys(data_in, bound=None, axis=None, axesdata=None, update_axis=True):
    """
    obtain a subarray of the data array

    :param data_in: array_like
    :param bound: tuple that marks the min and max of axes range after this function
    :param axis: int that marks the axis number corresponds to range, assume to be the last dimension if not specified
    :param axesdata: change the axesdata accordingly
    :param update_axis: if true update axesdata
    :return: hdf_data or (data, axes)
    """
    if isinstance(data_in, hdf_data):
        data = data_in.data
        axesdata = data_in.axes
    else:
        data = data_in
    if (not axesdata) or (not bound):
        return data_in
    if not axis:
        axis = axesdata[-1].axis_number
    inds, inde = [0, ] * len(axesdata), np.array(data.shape)
    ax = axesdata[axis]
    inds[axis] = int(round((bound[0] - ax.axis_min) / ax.increment))
    if inds[axis] < 0:
        inds[axis] = 0
    tmp = int(round((bound[1] - ax.axis_min) / ax.increment))
    if tmp < inde[axis]:
        inde[axis] = tmp
    inds, inde = tuple(inds), tuple(inde)
    if update_axis:
        data, axesdata = subrange(data, axesdata=axesdata, npts_start=inds, npts_end=inde)
    else:
        data = subrange(data, npts_start=inds, npts_end=inde)
    if isinstance(data_in, hdf_data):
        data_in.data = data
        if update_axis:
            data_in.axes = axesdata
        data_in.shape = data.shape
        return data_in
    if update_axis:
        return data, axesdata
    else:
        return data


def subrange(data_in, npts_start=None, npts_end=None, axesdata=None):
    """
    obtain a subarray of the data array

    :param data_in: array_like
    :param npts_start: int or ints marks starting index
    :param npts_end: int or ints marks ending index
    :param axesdata: change the axesdata accordingly
    :return: data and axesdata (if provided)
    """
    if isinstance(data_in, hdf_data):
        data = data_in.data
        axesdata = data_in.axes
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
                for j in range(pos, ndim):
                    if xmax == axesdata[j].axis_max:
                        pos = j + 1
                        ind[i] = j
        # chcek if the range is legit
        if npts_start < npts_end <= data.shape:
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
            print("illegal subrange, do nothing.")
        if axesdata:
            for di in range(ndim):
                axesdata[di].axis_min += axesdata[di].increment * npts_start[di]
                axesdata[di].axis_numberpoints = npts_end[di] - npts_start[di]
                axesdata[di].axis_max = (axesdata[di].increment * axesdata[di].axis_numberpoints +
                                         axesdata[di].axis_min)
    if isinstance(data_in, hdf_data):
        data_in.data = data
        data_in.axes = axesdata
        data_in.shape = data.shape
        if xdim == 2:
            data_in.XMAX = np.array([axesdata[ind[0]].axis_max, axesdata[ind[1]].axis_max])
        elif xdim == 1:
            data_in.XMAX = np.array([axesdata[ind[0]].axis_max])
        elif xdim == 3:
            data_in.XMAX = np.array([axesdata[ind[0]].axis_max, axesdata[ind[1]].axis_max, axesdata[ind[2]].axis_max])
        return data_in
    if axesdata:
        return data, axesdata
    else:
        return data


def adjust(data_obj, ops_list):
    """
    Analysis data and change axes accordingly

    :param data_obj: hdf_data
    :param ops_list: list of operations (str2keywords objects or strings)
    :return: return processed data
    """
    if not isinstance(data_obj, hdf_data):
        raise TypeError('Expecting hdf_data')
    for op in ops_list:
        if isinstance(op, str):
            op = str2keywords.str2keywords(op)
        if op == 'subrange':
            data_obj = subrange(data_obj, **op.keywords)
        elif op == 'subrange_phys':
            data_obj = subrange_phys(data_obj, **op.keywords)
    return data_obj
