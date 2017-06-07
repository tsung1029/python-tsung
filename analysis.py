import numpy as np
import str2keywords
from h5_utilities import *
from scipy.signal import hilbert


def update_fft_axes(axes, forward=True):
    if forward:
        print 'forward transform'
    else:
        print 'backward transform'
    return axes


def analysis(data, ops_list, axes=None):
    """
    Analysis data and change axes accordingly
    
    :param data: array_like data
    :param ops_list: list of operations (str2keywords objects)
    :param axes: list of axes (data_basic_axis objects) pass only the axes that need changes
    :return: return processed data (and axes if provided)  
    """
    for op in ops_list:
        if op == 'abs':
            data = np.abs(data)
        elif op == 'log':
            data = np.log(np.abs(data) + 1e-11)
        elif op == 'log10':
            data = np.log10(np.abs(data) + 1e-11)
        elif op == 'square':
            data = np.square(data)
        elif op == 'sqrt':
            data = np.sqrt(data)
        elif op == 'hilbert_env':
            data = np.abs(hilbert(data, **op.keywords))
        elif op == 'fft':
            ax = op.keywords.get('axes', None)
            data = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(data, axes=ax), **op.keywords), axes=ax)
        elif op == 'ifft':
            ax = op.keywords.get('axes', None)
            data = np.fft.ifftshift(np.fft.ifftn(np.fft.fftshift(data, axes=ax), **op.keywords), axes=ax)
    if axes:
        return data, axes
    else:
        return data

# tests
if __name__ == '__main__':
    kw = str2keywords.str2keywords('square')
    a = np.mgrid[:3, :3][0]
    # pass in list of keywords
    a = analysis(a, [kw])
    print a
