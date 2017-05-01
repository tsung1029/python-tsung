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

e2=sorted(glob.glob(dirname+'/MS/FLD/e2/*.h5'))
e3=sorted(glob.glob(dirname+'/MS/FLD/e3/*.h5'))
b2=sorted(glob.glob(dirname+'/MS/FLD/b2/*.h5'))
b3=sorted(glob.glob(dirname+'/MS/FLD/b3/*.h5'))
total_time=len(e2)
my_share = total_time/size
i_begin=rank*my_share
if (rank < (size -1)):
  i_end = (rank+1)*my_share 
else:
  i_end = total_time 
part= total_time/size

h5_filename=e2[1]
h5_data=read_hdf(h5_filename)
array_dims=h5_data.shape
nx=array_dims[0]
ny=array_dims[1]
time_step=h5_data.run_attributes['TIME'][0]
h5_output=hdf_data()
h5_output.shape=[total_time,ny]
print 'nx='+repr(nx)
print 'ny='+repr(ny)
print 'time_step='+repr(time_step)
print 'total_time='+repr(total_time)
h5_output.data=numpy.zeros((total_time,ny))
if (rank == 0):
  sum=numpy.zeros((total_time,ny))
h5_output.axes=[data_basic_axis(0,h5_data.axes[0].axis_min,h5_data.axes[0].axis_max,ny),
data_basic_axis(1,0.0,(time_step*total_time-1),total_time)]
h5_output.run_attributes['TIME']=0.0
h5_output.run_attributes['UNITS']='m_e /T'
h5_output.axes[0].attributes['LONG_NAME']=h5_data.axes[0].attributes['LONG_NAME']
h5_output.axes[0].attributes['UNITS']=h5_data.axes[0].attributes['UNITS']
h5_output.axes[1].attributes['LONG_NAME']='TIME'
h5_output.axes[1].attributes['UNITS']='1/\omega_p'

file_number=0
for file_number in range(i_begin,i_end):
  e2_filename=e2[file_number]
  e3_filename=e3[file_number]
  b2_filename=b2[file_number]
  b3_filename=b3[file_number]
  print e2_filename
  e2_data=read_hdf(e2_filename)
  e3_data=read_hdf(e3_filename)
  b2_data=read_hdf(b2_filename)
  b3_data=read_hdf(b3_filename)
  s1_data = e2_data * b3_data - e3_data * b2_data
  temp=numpy.sum(s1_data,axis=0)/nx
  h5_output.data[file_number,1:ny]=temp[1:ny]
  # file_number+=1

comm.Reduce(h5_output.data,sum,op=MPI.SUM,root=0)
if (rank==0):
  h5_output.data=sum

print 'before write'
print outfilename
if (rank==0):
  write_hdf(h5_output,outfilename)

print 'after write'

comm.barrier()
