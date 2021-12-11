# *************************************************************************
#
# Combine 2D data into 1 HDF5 file
# for binned data
#
# ************************************************************************* 
# usage python combine_2d_list.py DIRNAME OUT_ROOT
# output = OUT_ROOT-xx.h5
# *************************************************************************
#
# (ver 2.0 -- here we handle binned data)
#
# *************************************************************************
#
# FST, (c) 2019 Regents of The University of California
#
# *************************************************************************


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


argc=len(sys.argv)
if(argc < 3):
    print('Usage: python combine_2d_list.py DIRNAME OUT_ROOT')
    sys.exit()

dirname=sys.argv[1]
outfilename = sys.argv[2]

filelist=sorted(glob.glob(dirname+'/*.h5'))
total_time=len(filelist)
my_share = total_time 
i_begin=0
i_end = total_time 
if i_end > 1000: i_end = 999

h5_filename=filelist[1]
h5_data=osh5io.read_h5(h5_filename)
list_length=len(h5_data)
array_dims=h5_data[0].shape
nx=array_dims[0]
ny=array_dims[1]
time_step=h5_data[0].run_attrs['TIME'][0]
# h5_output=hdf_data()
# h5_output.shape=[total_time,ny]
print('nx='+repr(nx))
print('time_step='+repr(time_step))
print('total_time='+repr(total_time))
#
xaxis=h5_data[0].axes[1]
taxis=osh5def.DataAxis(0, time_step * (total_time -1), total_time,
    attrs={'NAME':'t', 'LONG_NAME':'time', 'UNITS':'1 / \omega_p'})

data_attrs=h5_data[0].data_attrs


h5_output= np.zeros((list_length,total_time,ny))
# h5_output.axes=[data_basic_axis(0,h5_data[0].axes[0].min,h5_data[0].axes[0].max,ny),
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
  for i in range(list_length):
      temp=np.average((h5_data[i]),axis=0)
      h5_output[i,file_number,1:ny]=temp[1:ny]
  # file_number+=1



print('before write')
for i in range(list_length):
    list_filename="%s-%02d.h5" %(outfilename,i)
    print(list_filename)
    b=osh5def.H5Data(h5_output[i,:,:], timestamp='x', data_attrs=data_attrs,run_attrs=run_attrs, axes=[taxis, xaxis])
    osh5io.write_h5(b,filename=list_filename)
print('after write')
print('Before barrier'+repr(rank))
