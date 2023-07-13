#
# analysis.py
# by Frank S. Tsung
#
# most of these functions have been replaced by oshutils in pyVisOS.  
# please download and use that instead.
# 
import numpy as np
import str2keywords
from h5_utilities import hdf_data
from arraymask import mask
from scipy.signal import hilbert


def update_fft_axes(axisdata, forward=True):
    if forward:
        freq = np.fft.fftshift(np.fft.fftfreq(axisdata.axis_numberpoints, d=axisdata.increment))
        axisdata.increment = (freq[1] - freq[0]) * 2 * np.pi
        axisdata.axis_min, axisdata.axis_max = freq[0] * 2 * np.pi, freq[-1] * 2 * np.pi
        axisdata.attributes['NAME'][0] = 'K(' + axisdata.attributes['NAME'][0] + ')'
        axisdata.attributes['LONG_NAME'][0] = 'K(' + axisdata.attributes['LONG_NAME'][0] + ')'
        axisdata.attributes['UNITS'][0] = '1/(' + axisdata.attributes['UNITS'][0] + ')'
    else:
        ax = np.fft.ifftshift(np.fft.fftfreq(axisdata.axis_numberpoints, d=axisdata.increment))
        axisdata.increment = (ax[1] - ax[0]) * 2 * np.pi
        axisdata.axis_min, axisdata.axis_max = ax[0] * 2 * np.pi, ax[-1] * 2 * np.pi
        axisdata.attributes['NAME'][0] = axisdata.attributes['NAME'][0][2:-1]
        axisdata.attributes['LONG_NAME'][0] = axisdata.attributes['LONG_NAME'][0][2:-1]
        axisdata.attributes['UNITS'][0] = axisdata.attributes['UNITS'][0][3:-1]
    return axisdata


def update_rebin_axes(axisdata, fac, order='f'):
    if order == 'f' or order == 'F':
        factor = fac
    else:
        factor = fac[::-1]
    for i in range(len(axisdata)):
        axisdata[i].increment *= factor[i]
        axisdata[i].axis_numberpoints /= factor[i]
    return axisdata


# modified from SciPy cookbook
def rebin(a, fac):
    """
    rebin ndarray data into a smaller ndarray of the same rank whose dimensions
    are factors of the original dimensions. eg. An array with 6 columns and 4 rows
    can be reduced to have 6,3,2 or 1 columns and 4,2 or 1 rows.
    example usages:
     a=rand(6,4); b=rebin(a, fac=[3,2])
     a=rand(6); b=rebin(a, fac=[2])
    """
    shape = a.shape
    lenShape = len(shape)
    newshape = np.floor_divide(shape, np.asarray(fac))
    evList = ['a.reshape('] + \
             ['newshape[%d],fac[%d],'%(i,i) for i in range(lenShape)] + \
             [')'] + ['.sum(%d)'%(i+1) for i in range(lenShape)] + \
             ['/fac[%d]'%i for i in range(lenShape)]
    # print ''.join(evList)
    return eval(''.join(evList))


def analysis(data_obj, ops_list, axesdata=None):
    """
    Analysis data and change axes accordingly
    
    :param data_obj: array_like data or hdf_data
    :param ops_list: list of operations (str2keywords objects or strings)
    :param axesdata: list of axes (data_basic_axis objects)
    :return: return processed data (and axes if provided)  
    """
    if isinstance(data_obj, hdf_data):
        axesdata = data_obj.axes
        data = data_obj.data
    else:
        data = data_obj
    for op in ops_list:
        if isinstance(op, str):
            op = str2keywords.str2keywords(op)
        if op == 'abs':
            data = np.abs(data)
        elif op == 'log':
            data = np.log(np.abs(data) + np.finfo('float64').eps)
        elif op == 'log10':
            data = np.log10(np.abs(data) + np.finfo('float64').eps)
        elif op == 'square':
            data = np.square(data)
        elif op == 'sqrt':
            data = np.sqrt(data)
        elif op == 'hilbert_env':
            data = np.abs(hilbert(data, **op.keywords))
        elif op == 'fft':
            ax = op.keywords.get('axes', None)
            data = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(data, axes=ax), **op.keywords), axes=ax)
            if axesdata:
                for i, ax in enumerate(axesdata):
                    axesdata[i] = update_fft_axes(axesdata[i], forward=True)
        elif op == 'ifft':
            ax = op.keywords.get('axes', None)
            data = np.fft.ifftshift(np.fft.ifftn(np.fft.fftshift(data, axes=ax), **op.keywords), axes=ax)
            if axesdata:
                for i, ax in enumerate(axesdata):
                    axesdata[i] = update_fft_axes(axesdata[i], forward=False)
        elif op == 'mask':
            data = mask(data, **op.keywords)
        elif op == 'rebin':
            data = rebin(data, **op.keywords)
            if axesdata:
                axesdata = update_rebin_axes(axesdata, **op.keywords)
        else:
            # (*NOT SAFE*) advance user may call functions from any known libaries (only numpy for now)
            # ********* CAUTIONS *************
            # 1. be careful with the memory layout of data
            # 2. the function can only accept one input
            # 3. there is not way to modify axes data
            attrs = [x for x in dir(np) if '__' not in x]
            if op.id in attrs:
                data = getattr(np, op.id)(data, **op.keywords)
            else:
                print('You used ' + op.id + ' but nothing happened!')
    if isinstance(data_obj, hdf_data):
        data_obj.data = data
        data_obj.shape = data.shape
        data_obj.axes = axesdata
        return data_obj
    if axesdata:
        return data, axesdata
    else:
        return data


# tests
if __name__ == '__main__':
    kw = str2keywords.str2keywords('fft ')
    a = np.mgrid[:3, :3][0]
    # pass in list of keywords
    a = analysis(a, [kw])
    print(a)
