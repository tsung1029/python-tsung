from h5_utilities import *
# import multiprocessing as mp
import sys
import getopt
import glob
import numpy as np
from mpi4py import MPI


def print_help():
    print 'para_sum_dirs.py [options] <OutputDir> <InputDir1> <InputDir2> ...'
    print 'options:'
    print '  --sq: sum of squared values, default is naive sum'
    print '  --abs: sum of absolute values, default is naive sum'


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
try:
    opts, args = getopt.gnu_getopt(sys.argv[1:], "hxy", ['sq', 'abs'])
except getopt.GetoptError:
    print_help()
    sys.exit(2)

if len(args) < 2:
    print_help()
    sys.exit(2)
outdir, indirs = args[0], args[1:]
indirc = len(indirs)
# do not overwirte data by accident!
for i in xrange(indirc):
    if os.path.realpath(outdir) == os.path.realpath(indirs[i]):
        sys.exit('InputDir and OutputDir cannot be the same!')
if rank == 0:
    if not os.path.exists(outdir):
        os.makedirs(outdir)
comm.barrier()
if outdir[-1] != '/':
    outdir += '/'
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
# read file list only in head node
if rank == 0:
    fileList = [sorted(glob.glob(indirs[i] + '/*.h5')) for i in xrange(indirc)]
else:
    fileList = None
fileList = comm.bcast(fileList, root=0)

# divide the task
fc = [len(fileList[i]) for i in xrange(indirc)]
total_time = min(fc)
my_share = total_time / size
i_begin = rank * my_share
if rank < (size - 1):
    i_end = (rank + 1) * my_share
else:
    i_end = total_time
part = total_time / size


# for all outputs at the same time
def foreach_sum(file_num):
    tmp = 0
    for i in xrange(indirc):
        h5data = read_hdf(fileList[i][file_num])
        if sum_type == 'sq':
            tmp += (h5data.data * h5data.data)
        elif sum_type == '+':
            tmp += h5data.data
        else:  # '--abs'
            tmp += np.abs(h5data.data)
    h5data.data = tmp / indirc
    write_hdf(h5data, outdir + os.path.basename(fileList[0][file_num]))


if __name__ == '__main__':
    # pool = mp.Pool()
    # time_i = range(i_begin, i_end)
    # pool.map(foreach_sum, time_i)
    # pool.close()
    # pool.join()
    for time_i in xrange(i_begin, i_end):
        foreach_sum(time_i)

comm.barrier()
