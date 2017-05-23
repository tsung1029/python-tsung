from h5_utilities import *
import matplotlib.pyplot as plt
import sys
import getopt
import glob
import numpy
from mpi4py import MPI


def print_help():
    print 'para_poynt.py [options] <InputDir> <OutputName>'
    print 'options:'
    print '  -n: average over n grid points at entrance (32 by default)'
    print '  --avg: look into the -savg directory'
    print '  --env: look into the -senv directory'

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
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
n_avg = 32
sumdir = 1
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
        print print_help()
        sys.exit(2)

e2 = sorted(glob.glob(dirName + '/MS/FLD/e2' + dir_ext + '/*.h5'))
e3 = sorted(glob.glob(dirName + '/MS/FLD/e3' + dir_ext + '/*.h5'))
b2 = sorted(glob.glob(dirName + '/MS/FLD/b2' + dir_ext + '/*.h5'))
b3 = sorted(glob.glob(dirName + '/MS/FLD/b3' + dir_ext + '/*.h5'))
total_time = len(e2)
my_share = total_time / size
i_begin = rank * my_share
if rank < (size - 1):
    i_end = (rank + 1) * my_share
else:
    i_end = total_time
part = total_time / size

h5_filename = e2[1]
h5_data = read_hdf(h5_filename)
array_dims = h5_data.shape
nx = array_dims[0]
ny = array_dims[1]

time_step = h5_data.run_attributes['TIME'][0]
h5_output = hdf_data()
h5_output.shape = [total_time, ny]
print 'nx=' + repr(nx)
print 'ny=' + repr(ny)
print 'time_step=' + repr(time_step)
print 'total_time=' + repr(total_time)
h5_output.data = numpy.zeros((total_time, ny))
income = numpy.zeros((total_time, ny))

h5_output2 = hdf_data()
h5_output2.shape = [total_time, nx]
h5_output2.data = numpy.zeros((total_time, nx))
income2 = numpy.zeros((total_time, nx))
total = 0
total2 = 0
if rank == 0:
    total = numpy.zeros((total_time, ny))
    total = numpy.zeros((total_time, nx))
h5_output.axes = [data_basic_axis(0, h5_data.axes[0].axis_min, h5_data.axes[0].axis_max, ny),
                  data_basic_axis(1, 0.0, (time_step * total_time - 1), total_time)]
h5_output.run_attributes['TIME'] = 0.0
h5_output.run_attributes['UNITS'] = 'm_e /T'
h5_output.axes[0].attributes['LONG_NAME'] = h5_data.axes[0].attributes['LONG_NAME']
h5_output.axes[0].attributes['UNITS'] = h5_data.axes[0].attributes['UNITS']
h5_output.axes[1].attributes['LONG_NAME'] = 'TIME'
h5_output.axes[1].attributes['UNITS'] = '1/\omega_p'

h5_output2.axes = [data_basic_axis(0, h5_data.axes[0].axis_min, h5_data.axes[0].axis_max, ny),
                   data_basic_axis(1, 0.0, (time_step * total_time - 1), total_time)]
h5_output2.run_attributes['TIME'] = 0.0
h5_output2.run_attributes['UNITS'] = 'm_e /T'
h5_output2.axes[0].attributes['LONG_NAME'] = h5_data.axes[0].attributes['LONG_NAME']
h5_output2.axes[0].attributes['UNITS'] = h5_data.axes[0].attributes['UNITS']
h5_output2.axes[1].attributes['LONG_NAME'] = 'TIME'
h5_output2.axes[1].attributes['UNITS'] = '1/\omega_p'

file_number = 0
for file_number in range(i_begin, i_end):
    e2_filename = e2[file_number]
    e3_filename = e3[file_number]
    b2_filename = b2[file_number]
    b3_filename = b3[file_number]
    print e2_filename
    e2_data = read_hdf(e2_filename)
    e3_data = read_hdf(e3_filename)
    b2_data = read_hdf(b2_filename)
    b3_data = read_hdf(b3_filename)
    s1_data = e2_data.data * b3_data.data - e3_data.data * b2_data.data
    temp = numpy.sum(s1_data, axis=0) / nx
    h5_output.data[file_number, 1:ny] = temp[1:ny]
    temp = numpy.sum(s1_data[1:n_avg+1], axis=0) / n_avg
    income[file_number, 1:ny] = temp[1:ny]
    temp = numpy.sum(s1_data, axis=1) / ny
    h5_output2.data[file_number, 1:nx] = temp[1:nx]
    temp = numpy.sum(s1_data[1:n_avg+1], axis=1) / n_avg
    income2[file_number, 1:nx] = temp[1:nx]
    # file_number+=1

comm.Reduce(h5_output.data, total, op=MPI.SUM, root=0)
if rank == 0:
    h5_output.data = total
    write_hdf(h5_output, outFilename)
comm.barrier()
comm.Reduce(income, total, op=MPI.SUM, root=0)
if rank == 0:
    h5_output.data = total
    newName = outFilename.rsplit('.', 1)[0] + '-x-n' + str(n_avg) + '.h5'
    write_hdf(h5_output, newName)

comm.Reduce(h5_output2.data, total2, op=MPI.SUM, root=0)
if rank == 0:
    h5_output2.data = total2
    write_hdf(h5_output2, outFilename)
comm.barrier()
comm.Reduce(income2, total2, op=MPI.SUM, root=0)
if rank == 0:
    h5_output2.data = total2
    newName = outFilename.rsplit('.', 1)[0] + '-y-n' + str(n_avg) + '.h5'
    write_hdf(h5_output2, newName)
