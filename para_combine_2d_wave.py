
#
# Combine 2d data into 1 HDF file
# this script is meant for quantities that 
# are wave-like.  In this case we take advantage of Parseval's theorem
# and sum the squares of the data.
#
#
# FST, (c) 2019 Regents of The University of California
#


import sys
sys.path.append('/Users/franktsung/Documents/codes/python-tsung/')
sys.path.append('/Volumes/Lacie-5TB/codes/pyVisOS/')

import osh5io
import osh5def
import osh5vis
import osh5utils

from h5_utilities import *
import matplotlib.pyplot as plt
import sys
import getopt
import glob
import numpy as np

# *****************************************************************************
# *****************************************************************************
# MPI related
import mpi4py
from mpi4py import MPI
# MPI related
# *****************************************************************************
# *****************************************************************************


def print_help():
    print('mpirun -n X python para_combine_2d_wave.py [options] <InputDir> <OutputName>')
    print('options:')
    print('  -n: average over n grid points at entrance (32 by default)')
    print('  -s: only process every S files')
    print('  --avg: look into the -savg directory')
    print('  --env: look into the -senv directory')

# *****************************************************************************
# *****************************************************************************
# MPI initialization
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
# *****************************************************************************
# *****************************************************************************
# *****************************************************************************



argc=len(sys.argv)
if(argc < 3):
    print('Usage: mpirun -np X python para_combine_2d_wave.py DIRNAME OUTNAME')
    sys.exit()

dirname=sys.argv[1]
outfilename = sys.argv[2]

filelist=sorted(glob.glob(dirname+'/*.h5'))
total_time=len(filelist)
my_share = total_time 
i_begin=0
i_end = total_time 

h5_filename=filelist[1]
h5_data=osh5io.read_h5(h5_filename)
array_dims=h5_data.shape
nx=array_dims[0]
ny=array_dims[1]
time_step=h5_data.run_attrs['TIME'][0]
# h5_output=hdf_data()
# h5_output.shape=[total_time,ny]
print('nx='+repr(nx))
print('time_step='+repr(time_step))
print('total_time='+repr(total_time))
#
xaxis=h5_data.axes[1]
taxis=osh5def.DataAxis(0, time_step * (total_time -1), total_time,
    attrs={'NAME':'t', 'LONG_NAME':'time', 'UNITS':'1 / \omega_p'})

data_attrs=h5_data.data_attrs


h5_output= np.zeros((total_time,ny))
total    = np.zeros((total_time,ny))
# h5_output.axes=[data_basic_axis(0,h5_data.axes[0].min,h5_data.axes[0].max,ny),
# data_basic_axis(1,0.0,(time_step*total_time-1),total_time)]


# run_attrs = h5_output.run_attrs
run_attrs = {'XMAX' : np.array( [ time_step * (total_time-1),xaxis.max] ) , 
            'XMIN' : np.array( [0, xaxis.min, 0] ) }


file_number=0
for file_number in range(i_begin,i_end):
  h5_filename=filelist[file_number]
  if (file_number % 10 == 0 ):
      print(h5_filename)
  h5_data=osh5io.read_h5(h5_filename)
  h5_data = h5_data**2
  temp=np.sum((h5_data.data),axis=0)  ! sum the squares of the data, store it below
  # temp=np.sqrt(temp)
  h5_output[file_number,1:ny]=temp[1:ny]
  # file_number+=1



print('before write')
print(outfilename)
b=osh5def.H5Data(h5_output, timestamp='x', data_attrs=data_attrs,run_attrs=run_attrs, axes=[taxis, xaxis])
osh5io.write_h5(b,filename=outfilename)
print('after write')
print('Before barrier'+repr(rank))
