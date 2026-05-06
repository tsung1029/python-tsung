import sys
import glob
import numpy as np

# Modify sys.path for osh5io and osh5def if needed
sys.path.append('/Users/franktsung/Documents/codes/python-tsung/')
sys.path.append('/Volumes/Lacie-5TB/codes/pyVisOS/')

import osh5io
import osh5def

def combine_2d(dirname, outfilename):
    """
    Combine 2D data into 1 HDF5 file
    the data is averaged in the x2 direction
    """
    filelist = sorted(glob.glob(dirname + '/*.h5'))
    total_time = len(filelist)
    
    if total_time == 0:
        print('No .h5 files found in the directory.')
        return

    i_begin = 0
    i_end = total_time 

    # Read the first file to get dimensions and attributes
    h5_filename = filelist[0]
    h5_filename_prime = filelist[1]
    h5_data = osh5io.read_h5(h5_filename)
    h5_data_prime=osh5io.read_h5(h5_filename_prime)
    array_dims = h5_data.shape

    nx = array_dims[0]
    ny = array_dims[1]
    time_step = h5_data_prime.run_attrs['TIME'][0]-h5_data.run_attrs['TIME'][0]

    print('nx=' + repr(nx))
    print('time_step=' + repr(time_step))
    print('total_time=' + repr(total_time))

    xaxis = h5_data.axes[1]
    taxis = osh5def.DataAxis(0, time_step * (total_time - 1), total_time,
        attrs={'NAME': 't', 'LONG_NAME': 'time', 'UNITS': '1 / \\omega_p'})

    data_attrs = h5_data.data_attrs

    h5_output = np.zeros((total_time, ny))
    
    run_attrs = {'XMAX' : np.array([time_step * (total_time - 1), xaxis.max]), 
                 'XMIN' : np.array([0, xaxis.min, 0])}

    for file_number in range(i_begin, i_end):
        h5_filename = filelist[file_number]
        if file_number % 10 == 0:
            print(h5_filename)
        try:
            h5_data_old = h5_data
            h5_data = osh5io.read_h5(h5_filename)
        except OSError:
            h5_data = h5_data_old

        temp = np.average(h5_data.data, axis=0)
        h5_output[file_number, 1:ny] = temp[1:ny]

    print('before write')
    print(outfilename)
    b = osh5def.H5Data(h5_output, timestamp='x', data_attrs=data_attrs, run_attrs=run_attrs, axes=[taxis, xaxis])
    osh5io.write_h5(b, filename=outfilename)
    print('after write')


def combine_1d(dirname, outfilename):
    """
    Combine 1D data into 1 HDF5 file
    """
    filelist = sorted(glob.glob(dirname + '/*.h5'))
    total_time = len(filelist)
    
    if total_time == 0:
        print('No .h5 files found in the directory.')
        return

    i_begin = 0
    i_end = total_time 

    # Read the first file to get dimensions and attributes
    h5_filename = filelist[0]
    h5_data = osh5io.read_h5(h5_filename)
    array_dims = h5_data.shape
    nx = array_dims[0]
    time_step = h5_data.run_attrs['TIME'][0]

    print('nx=' + repr(nx))
    print('time_step=' + repr(time_step))
    print('total_time=' + repr(total_time))

    xaxis = h5_data.axes[0]
    taxis = osh5def.DataAxis(0, time_step * (total_time - 1), total_time,
        attrs={'NAME': 't', 'LONG_NAME': 'time', 'UNITS': '1 / \\omega_p'})

    data_attrs = h5_data.data_attrs

    h5_output = np.zeros((total_time, nx))
    
    run_attrs = {'XMAX' : np.array([time_step * (total_time - 1), xaxis.max]), 
                 'XMIN' : np.array([0, xaxis.min, 0])}

    for file_number in range(i_begin, i_end):
        h5_filename = filelist[file_number]
        if file_number % 10 == 0:
            print(h5_filename)
        h5_data = osh5io.read_h5(h5_filename)
        h5_output[file_number, 1:nx] = h5_data.data[1:nx]

    print('before write')
    print(outfilename)
    b = osh5def.H5Data(h5_output, timestamp='x', data_attrs=data_attrs, run_attrs=run_attrs, axes=[taxis, xaxis])
    osh5io.write_h5(b, filename=outfilename)
    print('after write')
