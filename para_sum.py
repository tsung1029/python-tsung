from h5_utilities import *
import multiprocessing as mp
import sys
import getopt
import glob
import numpy
from mpi4py import MPI


def print_help():
    print 'para_sum.py [options] <InputDir> <OutputName>'
    print 'options:'
    print '  --sq: sum of squared values (default)'
    print '  --abs: sum of absolute values'
    print '  -x: sum over x'
    print '  -y: sum over y (default)'

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
argc = len(sys.argv)
try:
    opts, args = getopt.gnu_getopt(sys.argv[1:], "hxy", ['sq', 'abs'])
except getopt.GetoptError:
    print_help()
    sys.exit(2)

if len(args) < 2:
    print_help()
    sys.exit(2)
dirName = args[0]
outFilename = args[1]

do_sq_sum = True
sum_in_y = True
for opt, arg in opts:
    if opt == '-h':
        print_help()
        sys.exit()
    elif opt == '-x':
        sum_in_y = False
    elif opt == '--abs':
        do_sq_sum = False

fileList = sorted(glob.glob(dirName + '/*.h5'))
total_time = len(fileList)
my_share = total_time / size
i_begin = rank * my_share
if rank < (size - 1):
    i_end = (rank + 1) * my_share
else:
    i_end = total_time
part = total_time / size

h5_filename = fileList[1]
h5_data = read_hdf(h5_filename)
array_dims = h5_data.shape
nx = array_dims[0]
ny = array_dims[1]
time_step = h5_data.run_attributes['TIME'][0]
h5_output = hdf_data()
if sum_in_y:
    nd = ny
    nds = nx
    idx = 0
else:
    nd = nx
    nds = ny
    idx = 1
h5_output.shape = [total_time, nd]
print 'nx=' + repr(nx)
print 'ny=' + repr(ny)
print 'time_step=' + repr(time_step)
print 'total_time=' + repr(total_time)
h5_output.data = numpy.zeros((total_time, nd))
# if rank == 0:
total = numpy.zeros((total_time, nd))
h5_output.axes = [data_basic_axis(0, h5_data.axes[idx].axis_min, h5_data.axes[idx].axis_max, nd),
                  data_basic_axis(1, 0.0, (time_step * total_time - 1), total_time)]
h5_output.run_attributes['TIME'] = 0.0
# h5_output.run_attributes['UNITS']=h5_data.run_attributes['UNITS']
h5_output.axes[0].attributes['LONG_NAME'] = h5_data.axes[idx].attributes['LONG_NAME']
h5_output.axes[0].attributes['UNITS'] = h5_data.axes[idx].attributes['UNITS']
h5_output.axes[1].attributes['LONG_NAME'] = 'TIME'
h5_output.axes[1].attributes['UNITS'] = '1/\omega_p'


# file_number = 0
# for file_number in range(i_begin, i_end):
def foreach_sum(file_num):
    h5_fname = fileList[file_num]
    print h5_fname
    h5data = read_hdf(h5_fname)
    if do_sq_sum:
        temp = numpy.sum((h5data.data * h5data.data), axis=idx) / nds
    else:  # '--sq'
        temp = numpy.sum(numpy.abs(h5data.data), axis=idx) / nds
    return temp[1:nd]

if __name__ == '__main__':
    pool = mp.Pool()
    time_i = range(i_begin, i_end)
    h5_output.data[time_i, 1:nd] = numpy.array(pool.map(foreach_sum, time_i))
    pool.close()
    pool.join()

comm.Reduce(h5_output.data, total, op=MPI.SUM, root=0)
if rank == 0:
    h5_output.data = total

# print 'before write'
if rank == 0:
    print 'yo, I am writing:   ' + outFilename
    write_hdf(h5_output, outFilename)

# print 'after write'

comm.barrier()
