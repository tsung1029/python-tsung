from h5_utilities import read_hdf, write_hdf
import arraymask
# import multiprocessing as mp
import str2keywords
import sys
import os
import getopt
import glob
import numpy as np
import analysis
import adjust
from mpi4py import MPI


def print_help():
    print 'para_pha_f_slope.py [options] <InputDir> <OutputDir>'
    print 'options:'
    print '  --dfdp: calculate df/dp instead of (df/dp)/(-p*f)'
    print '  --dim={1|2|3}: finite difference in which dimension'
    print '  --rebin=[binx, biny, ...]: array_like, number of bins along each dimension'
    print '  --adjust=string: adjust the data before rebinning. ' \
          '         String will be convert to str2keywords and call functions in adjust.py'


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
argc = len(sys.argv)
# # READ COMMAND LINE OPTIONS
try:
    opts, args = getopt.gnu_getopt(sys.argv[1:], "h:", ['dfdp', 'dim=', 'rebin=', 'adjust='])
except getopt.GetoptError:
    print_help()
    sys.exit(2)

if len(args) < 2:
    print_help()
    sys.exit(2)
dirName = args[0]
outdir = args[1]
# do not overwirte data by accident!
if os.path.realpath(dirName) == os.path.realpath(outdir):
    sys.exit('InputDir and OutputDir cannot be the same!')
if rank == 0:
    if not os.path.exists(outdir):
        os.makedirs(outdir)
if outdir[-1] != '/':
    outdir += '/'
# # SET DEFAULT OPTIONS
dfdp, dim, rebin = False, None, False
for opt, arg in opts:
    if opt == '-h':
        print_help()
        sys.exit()
    elif opt == '--dfdp':
        dfdp = True
    elif opt == '--dim':
        dim = eval(arg)
    elif opt == '--rebin':
        rebin = eval(arg)
    elif opt == '--adjust':
        adjust_ops = str2keywords.str2keywords(arg)
    else:
        print print_help()
        sys.exit(2)

# # GET DATA STREAMS
if rank == 0:
    flst = sorted(glob.glob(dirName + '/*.h5'))
else:
    flst = None
flst = comm.bcast(flst, root=0)

# # divide the task
total_time = len(flst)
my_share = total_time / size
i_begin = rank * my_share
if rank < (size - 1):
    i_end = (rank + 1) * my_share
else:
    i_end = total_time
part = total_time / size

# # READ ONE FILE, FIGURE OUT THE AXES INFORMATION AND SET COMMON PARAMETERS
h5_filename = flst[i_begin]
h5_output = read_hdf(h5_filename)
nx = h5_output.shape
ndim = h5_output.data.ndim

# make some adjustments before processing data, useful for rebinning etc.
if adjust_ops == 'subrange':
    h5_output = adjust.subrange(h5_output, axes=h5_output.axes, **adjust_ops.keywords)

# determine which dimension to differentiate
if not dim:
    for dim, ax in enumerate(h5_output.axes):
        if 'p' in ax.attributes['NAME'][0].lower():
            break
    else:
        sys.exit('Cannot find velocity axis and no dim specified. Exiting...')
h5_output.NAME[0] += ' slope'
# check if the origin is in the axis points to avoid divide by zero error
paxis = h5_output.axes[dim].get_axis_points()
if rebin:
    h5_output.axes = analysis.update_rebin_axes(h5_output.axes, fac=rebin)
# prepare a view of the axis data for later processing. this should work with ndarrays
view_arr = np.ones((1, h5_output.data.ndim), int).ravel()
# the axis numbering is in fortran order
view_arr[-(dim+1)] = -1
paxis = paxis.reshape(view_arr)


# # FOR EACH TIME STAMP DO SOMETHING
def foreach_decompose(file_num):
    f_filename = flst[file_num]
    ffile = read_hdf(f_filename)
    if adjust_ops == 'subrange':
        ffile.data = adjust.subrange(ffile.data, **adjust_ops.keywords)
    if rebin:
        ffile.data = analysis.rebin(ffile.data, fac=rebin)
    # central difference for interior, forward and backward at the boundaries
    grad = np.gradient(ffile.data, axis=((dim + ndim - 1) % ndim))
    if dfdp:
        h5_output.data = grad
    else:
        pf = - paxis * ffile.data
        # there are zeros because: 1. the origin is included in the axis points; 2. the tail of distribution is zero
        pf[pf == 0.] = 999999.0
        h5_output.data = np.divide(grad, pf)
    h5_output.run_attributes['TIME'][0] = ffile.run_attributes['TIME'][0]
    h5_output.run_attributes['ITER'][0] = ffile.run_attributes['ITER'][0]
    newname = outdir + os.path.basename(f_filename)
    write_hdf(h5_output, newname)
    return f_filename


# TODO: using mpi4py and multiprocess simutaneously may fail on some clusters. fix needed
if __name__ == '__main__':
    # # threaded version
    # if size == 1:
    #     pool = mp.Pool()
    #     print pool.map(foreach_decompose, range(i_begin, i_end))
    #     pool.close()
    #     pool.join()
    # else:  # serial/mpi version
    #     for time_i in range(i_begin, i_end):
    #         foreach_decompose(time_i)
    for time_i in range(i_begin, i_end):
        foreach_decompose(time_i)

comm.Barrier()
comm.Disconnect()
