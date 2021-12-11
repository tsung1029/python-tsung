import sys

# including my path and Han Wen's path
sys.path.append('/Users/franktsung/Documents/codes/python-tsung/')
sys.path.append('/Volumes/Lacie-5TB/codes/pyVisOS/')
#


import osh5io
import osh5def
import osh5vis
import osh5utils

from h5_utilities import *
import matplotlib.pyplot as plt
import sys
# command line options (argc, argv stuff)
import getopt 
#

# glob -> finding files in a directory
import glob
#

import numpy as np


def print_help():
    print('para_poynt.py [options] <InputDir> <OutputName>')
    print('options:')
    print('  -n: average over n grid points at entrance (32 by default)')
    print('  --avg: look into the -savg directory')
    print('  --env: look into the -senv directory')

argc = len(sys.argv)
try:
    opts, args = getopt.gnu_getopt(sys.argv[1:], "hxyn:", ['avg', 'env'])
except getopt.GetoptError:
    print_help()
    sys.exit(2)

if len(args) < 2:
    print_help()
    sys.exit(2)
dirName = args[0]
outFilename = args[1]
dir_ext = ''
n_avg = 50
#
# sumdir = 0 summing over the transverse direction
sumdir = 0
#
#
tags = ''
for opt, arg in opts:
    if opt == '-h':
        print_help()
        sys.exit()
    elif opt == '--avg':
        dir_ext = '-savg'
    elif opt == '--env':
        dir_ext = '-senv'
    elif opt == '-n':
        n_avg = arg
    else:
        print(print_help())
        sys.exit(2)

e2 = sorted(glob.glob(dirName + '/FLD/e2' + dir_ext + '/*.h5'))
e3 = sorted(glob.glob(dirName + '/FLD/e3' + dir_ext + '/*.h5'))
b2 = sorted(glob.glob(dirName + '/FLD/b2' + dir_ext + '/*.h5'))
b3 = sorted(glob.glob(dirName + '/FLD/b3' + dir_ext + '/*.h5'))
total_time = len(e2)
my_share = total_time 
i_begin = 0
i_end = total_time -1
avg_array=np.ones(n_avg)/n_avg
#
# read the second file to get the time-step
#
h5_filename = e2[1]  
h5_data = osh5io.read_h5(h5_filename)
array_dims = h5_data.shape
nx = array_dims[0]
ny = array_dims[1]

time_step = h5_data.run_attrs['TIME'][0]
# h5_output = hdf_data()
# h5_output.shape = [total_time, nx]
print('nx=' + repr(nx))
print('ny=' + repr(ny))
print('time_step=' + repr(time_step))
print('total_time=' + repr(total_time))
h5_output = np.zeros((total_time, ny))
total = np.zeros((total_time,ny))

#total = 0
total2 = 0

xaxis=h5_data.axes[1]
taxis=osh5def.DataAxis(0, time_step * (total_time -1), total_time,
    attrs={'NAME':'t', 'LONG_NAME':'time', 'UNITS':'1 / \omega_p'})

data_attrs = { 'UNITS': osh5def.OSUnits('m_e \omega_p^3'), 'NAME': 's1', 'LONG_NAME': 'S_1' }

print(repr(xaxis.min))
print(repr(xaxis.max))
run_attrs = {'XMAX' : np.array( [time_step * (total_time-1), xaxis.max] ) , 
            'XMIN' : np.array( [0, xaxis.min ] ) }

# h5_output.run_attrs['TIME'] = 0.0
# h5_output.run_attrs['UNITS'] = 'm_e /T'



file_number = 0
skip = 1
for file_number in range(i_begin, i_end,skip):
    print(file_number)
    e2_filename = e2[file_number]
    e3_filename = e3[file_number]
    b2_filename = b2[file_number]
    b3_filename = b3[file_number]
    print(e2_filename)
    e2_data = osh5io.read_h5(e2_filename)
    e3_data = osh5io.read_h5(e3_filename)
    b2_data = osh5io.read_h5(b2_filename)
    b3_data = osh5io.read_h5(b3_filename)
    s1_data = e2_data * b3_data - e3_data * b2_data
    #if(file_number % 10 == 0):
        #print(s1_data.shape)
    temp=np.sum(s1_data,axis=0) / nx
    print(temp.shape)
    h5_output[file_number, 1:ny] = np.abs(np.convolve(temp[1:ny],avg_array,mode='same'))
    # h5_output[file_number, 1:nx] = np.convolve(s1_data[1:nx],avg_array,mode='same')
#    temp = np.sum(s1_data, axis=0) / nx
#    h5_output.data[file_number, 1:ny] = temp[1:ny]
#    temp = np.sum(s1_data[0:n_avg, :], axis=0) / n_avg
#    income[file_number, 1:ny] = temp[1:ny]
#    temp = np.sum(s1_data, axis=1) / ny
#    h5_output2.data[file_number, 1:nx] = temp[1:nx]
#    temp = np.sum(s1_data[:, 0:n_avg], axis=1) / n_avg
#    income2[file_number, 1:nx] = temp[1:nx]
    # file_number+=1

b=osh5def.H5Data(h5_output, timestamp='x', data_attrs=data_attrs,run_attrs=run_attrs, axes=[taxis, xaxis])
osh5io.write_h5(b,filename=outFilename)
#    write_hdf(h5_output, outFilename)
