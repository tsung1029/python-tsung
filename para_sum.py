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
    print '  --sq: sum of squared values, default is naive sum'
    print '  --abs: sum of absolute values, default is naive sum'
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
dirName, outFilename = args[0], args[1]

sum_type, sum_in_y = '+', True
for opt, arg in opts:
    if opt == '-h':
        print_help()
        sys.exit()
    elif opt == '-x':
        sum_in_y = False
    elif opt == '--sq':
        sum_type = 'sq'
    elif opt == '--abs':
        sum_type = 'abs'

fileList = sorted(glob.glob(dirName + '/*.h5'))
total_time = len(fileList)
my_share = total_time / size
i_begin = rank * my_share
if rank < (size - 1):
    i_end = (rank + 1) * my_share
else:
    i_end = total_time
part = total_time / size

h5_filename = fileList[0]
h5_data = read_hdf(h5_filename)
nx = h5_data.shape
t0 = h5_data.run_attributes['TIME'][0]
h5_data = read_hdf(fileList[1])
t1 = h5_data.run_attributes['TIME'][0]
h5_output = hdf_data()
if sum_in_y:
    idx = 0
else:
    idx = 1
nd = nx[-(idx+1)]

h5_output.data = numpy.zeros((total_time, nd))
h5_output.shape = h5_output.data.shape
total = numpy.zeros((total_time, nd))
h5_output.axes = [data_basic_axis(1, t0, (t1 - t0) * (total_time - 1), total_time), h5_data.axes[-(idx+1)]]
h5_output.run_attributes['TIME'] = 0.0
h5_output.axes[0].attributes['LONG_NAME'] = 'TIME'
h5_output.axes[0].attributes['UNITS'] = '1/\omega_p'


# file_number = 0
# for file_number in range(i_begin, i_end):
def foreach_sum(file_num):
    h5_fname = fileList[file_num]
    # print h5_fname
    h5data = read_hdf(h5_fname)
    if sum_type == 'sq':
        temp = numpy.sum((h5data.data * h5data.data), axis=idx)
    elif sum_type == '+':
        temp = numpy.sum(h5data.data, axis=idx)
    else:  # '--abs'
        temp = numpy.sum(numpy.abs(h5data.data), axis=idx)
    return temp


if __name__ == '__main__':
    pool = mp.Pool()
    time_i = range(i_begin, i_end)
    h5_output.data[time_i, :] = numpy.array(pool.map(foreach_sum, time_i))
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
