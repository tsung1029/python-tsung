
#
# Combine 1D data into 1 HDF5 file
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
from mpi4py import MPI


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
argc=len(sys.argv)
if(argc < 3):
    print('Usage: python para_combine_2d.py DIRNAME OUTNAME')
    sys.exit()

dirname=sys.argv[1]
outfilename = sys.argv[2]

filelist=sorted(glob.glob(dirname+'/*.h5'))
total_time=len(filelist)
my_share = total_time // size
i_begin=rank*my_share
if (rank < (size -1)):
  i_end = (rank+1)*my_share -1
else:
  i_end = total_time - 1
part= total_time/size

h5_filename=filelist[1]
h5_data=osh5io.read_h5(h5_filename)
array_dims=h5_data.shape
nx=array_dims[0]
# ny=array_dims[1]
time_step=h5_data.run_attributes['TIME'][0]
h5_output=hdf_data()
h5_output.shape=[total_time,nx]
print('nx='+repr(nx))
print('time_step='+repr(time_step))
print('total_time='+repr(total_time))

h5_output.data=numpy.zeros((total_time,nx))
if (rank == 0):
  sum=numpy.zeros((total_time,nx))
h5_output.axes=[data_basic_axis(0,h5_data.axes[0].axis_min,h5_data.axes[0].axis_max,ny),
data_basic_axis(1,0.0,(time_step*total_time-1),total_time)]


data_attrs = h5_output.data_attrs
run_attrs = {'XMAX' : np.array( [ time_step * (total_time-1),xaxis.max] ) , 
            'XMIN' : np.array( [0, xaxis.min, 0] ) }


file_number=0
for file_number in range(i_begin,i_end):
  h5_filename=filelist[file_number]
  if (i_count % 10 == 0 and rank == 0):
      print(h5_filename)
  h5_data=osh5io.read_h5(h5_filename)
  temp=numpy.sum(numpy.abs(h5_data.data),axis=0)/nx
  h5_output[file_number,1:nx]=temp[1:nx]
  # file_number+=1

comm.Reduce(h5_output,total,op=MPI.SUM,root=0)


print('before write')
print(outfilename)
if (rank==0):
  b = osh5def.H5Data(total, timestamp='x', data_attrs=data_attrs,
                     run_attrs=run_attrs, axes=[taxis,xaxis])
  osh5io.write_h5(b,filename=outFilename)
print('after write')

comm.barrier()
