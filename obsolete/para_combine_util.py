from h5_utilities import *
import matplotlib.pyplot as plt
import sys
import glob
import numpy
from mpi4py import MPI


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
argc=len(sys.argv)
if(argc < 3):
    print('Usage: python 1d_combine.py DIRNAME OUTNAME')
    sys.exit()

dirname=sys.argv[1]
outfilename = sys.argv[2]

filelist=sorted(glob.glob(dirname+'/*.h5'))
total_time=len(filelist)
my_share = total_time // size
i_begin=rank*my_share
if (rank < (size -1)):
  i_end = (rank+1)*my_share
else:
  i_end = total_time 
part= total_time/size

h5_filename=filelist[1]
h5_data=read_hdf(h5_filename)
array_dims=h5_data.shape
nx=array_dims[0]
ny=array_dims[1]
time_step=h5_data.run_attributes['TIME'][0]
h5_output=hdf_data()
h5_output.shape=[total_time,ny]
print('nx='+repr(nx))
print('ny='+repr(ny))
print('time_step='+repr(time_step))
print('total_time='+repr(total_time))
h5_output.data=numpy.zeros((total_time,ny))
if (rank == 0):
  sum=numpy.zeros((total_time,ny))
h5_output.axes=[data_basic_axis(0,h5_data.axes[0].axis_min,h5_data.axes[0].axis_max,ny),
data_basic_axis(1,0.0,(time_step*total_time-1),total_time)]
h5_output.run_attributes['TIME']=0.0
# h5_output.run_attributes['UNITS']=h5_data.run_attributes['UNITS']
h5_output.axes[0].attributes['LONG_NAME']=h5_data.axes[0].attributes['LONG_NAME']
h5_output.axes[0].attributes['UNITS']=h5_data.axes[0].attributes['UNITS']
h5_output.axes[1].attributes['LONG_NAME']='TIME'
h5_output.axes[1].attributes['UNITS']='1/\omega_p'

file_number=0
for file_number in range(i_begin,i_end):
  h5_filename=filelist[file_number]
  print(h5_filename)
  h5_data=read_hdf(h5_filename)
  temp=numpy.sum(numpy.abs(h5_data.data),axis=0)/nx
  h5_output.data[file_number,1:ny]=temp[1:ny]
  # file_number+=1

comm.Reduce(h5_output.data,sum,op=MPI.SUM,root=0)
if (rank==0):
  h5_output.data=sum

print('before write')
print(outfilename)
if (rank==0):
  write_hdf(h5_output,outfilename)
print('after write')

comm.barrier()