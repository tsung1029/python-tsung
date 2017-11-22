from h5_utilities import read_hdf, write_hdf
import arraymask
# import multiprocessing as mp
import sys
import os
import getopt
import glob
import numpy as np
import analysis
from mpi4py import MPI


def print_help():
    print 'para_decompose_2d.py [options] <InputDir> <OutputDir>'
    print 'options:'
    print '  --avg: look into the -savg directory'
    print '  --env: look into the -senv directory'
    print '  --blockx=[lower, upper]: set kx between to be zeros'
    print '  --blocky=[lower, upper]: set ky between to be zeros'
    print '  --ifftdim=[x1,x2]: which direction for inverse fourier transform'
    print '  --ocomp=[l,t1,t2,t]: which component to output'


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
argc = len(sys.argv)
# # READ COMMAND LINE OPTIONS
try:
    opts, args = getopt.gnu_getopt(sys.argv[1:], "h:", ['avg', 'env', 'blockx=', 'blocky=', 'ifftdim=', 'ocomp='])
except getopt.GetoptError:
    print_help()
    sys.exit(2)

if len(args) < 2:
    print_help()
    sys.exit(2)
dirName = args[0]
outdir = args[1]
if rank == 0:
    if not os.path.exists(outdir):
        os.makedirs(outdir)
if outdir[-1] != '/':
    outdir += '/'
# # SET DEFAULT OPTIONS
dir_ext, blockx, blocky, ifftdim, ocomp = '', None, None, None, ['l', 't']
for opt, arg in opts:
    if opt == '-h':
        print_help()
        sys.exit()
    elif opt == '--avg':
        dir_ext = '-savg'
    elif opt == '--env':
        dir_ext = '-senv'
    elif opt == '--blockx':
        blockx = eval(arg)
    elif opt == '--blocky':
        blocky = eval(arg)
    elif opt == '--ifftdim':
        ifftdim = arg.lstrip('[').rstrip(']').split(',')
    elif opt == '--ocomp':
        ocomp = arg.lower()
    else:
        print print_help()
        sys.exit(2)

# # GET DATA STREAMS
if rank == 0:
    elst = {'e1': sorted(glob.glob(dirName + '/MS/FLD/e1' + dir_ext + '/*.h5')),
            'e2': sorted(glob.glob(dirName + '/MS/FLD/e2' + dir_ext + '/*.h5'))}
else:
    elst = None
elst = comm.bcast(elst, root=0)
e1, e2 = elst['e1'], elst['e2']

# # divide the task
total_time = len(e1)
my_share = total_time / size
i_begin = rank * my_share
if rank < (size - 1):
    i_end = (rank + 1) * my_share
else:
    i_end = total_time
part = total_time / size

# # READ ONE FILE, FIGURE OUT THE AXES INFORMATION AND SET COMMON PARAMETERS
h5_filename = e1[i_begin]
h5_output = read_hdf(h5_filename)
nx = h5_output.shape
ky, kx = (analysis.update_fft_axes(h5_output.axes[0].clone(), forward=True),
          analysis.update_fft_axes(h5_output.axes[1].clone(), forward=True))
kxvec, kyvec = np.meshgrid(kx.get_axis_points(), ky.get_axis_points())
kamp = np.sqrt(np.square(kxvec) + np.square(kyvec))
# unit vector
kxvec, kyvec = np.nan_to_num(np.divide(kxvec, kamp)), np.nan_to_num(np.divide(kyvec, kamp))
if not ifftdim:
    ift = None
    h5_output.axes = [ky, kx]
else:
    ifftdim = np.array(ifftdim)
    if 'x2' in ifftdim and 'x1' in ifftdim:
        ift = 'axes=[0,1]'
    elif 'x2' not in ifftdim:
        h5_output.axes[0] = ky
        ift = 'axes=[0]'
    else:  # 'x1' not in ifftdim
        h5_output.axes[1] = kx
        ift = 'axes=[1]'
# block part of the spectrum
brg = []
if blockx:
    try:
        for bi in blockx:
            brg.append(('x1', bi[0], bi[1]))
    except TypeError:
        brg.append(('x1', blockx[0], blockx[1]))
if blocky:
    try:
        for bi in blocky:
            brg.append(('x2', bi[0], bi[1]))
    except TypeError:
        brg.append(('x2', blocky[0], blocky[1]))


# # FOR EACH TIME STAMP DO SOMETHING
def foreach_decompose(file_num):
    e_filename = e1[file_num]
    e1file = read_hdf(e_filename)
    e2file = read_hdf(e2[file_num])
    e1data = analysis.analysis(e1file.data, ['fft'])
    e2data = analysis.analysis(e2file.data, ['fft'])
    h5_output.run_attributes['TIME'][0] = e1file.run_attributes['TIME'][0]
    h5_output.run_attributes['ITER'][0] = e1file.run_attributes['ITER'][0]
    if blockx or blocky:
        e1data = arraymask.mask(e1data, axes=[ky, kx], region=brg)
        e2data = arraymask.mask(e2data, axes=[ky, kx], region=brg)

    dive = np.multiply(kxvec, e1data) + np.multiply(kyvec, e2data)

    h5_output.data = np.multiply(kxvec, dive)
    # transverse part, longitudinal part
    if ift:
        tr1, e1data = (analysis.analysis(e1data - h5_output.data, ['ifft ' + ift]),
                       analysis.analysis(h5_output.data, ['ifft ' + ift]))
    else:
        tr1, e1data = e1data - h5_output.data, h5_output.data
    if 't1' in ocomp:
        h5_output.data = np.abs(tr1)
        h5_output.run_attributes['NAME'][0], h5_output.data_attributes['LONG_NAME'] = 'Et1', 'E_{t1}'
        newname = outdir + 'Et1' + os.path.basename(e_filename)[2:]
        write_hdf(h5_output, newname)

    h5_output.data = np.multiply(kyvec, dive)
    if ift:
        tr2, e2data = (analysis.analysis(e2data - h5_output.data, ['ifft ' + ift]),
                       analysis.analysis(h5_output.data, ['ifft ' + ift]))
    else:
        tr2, e2data = e2data - h5_output.data, h5_output.data
    if 't2' in ocomp:
        h5_output.data = np.abs(tr2)
        h5_output.run_attributes['NAME'][0], h5_output.data_attributes['LONG_NAME'] = 'Et2', 'E_{t2}'
        newname = outdir + 'Et2' + os.path.basename(e_filename)[2:]
        write_hdf(h5_output, newname)
    if 't' in ocomp:
        h5_output.data = np.sqrt(np.abs(np.square(tr1) + np.square(tr2)))
        h5_output.run_attributes['NAME'][0], h5_output.data_attributes['LONG_NAME'] = 'ET', 'E_T'
        newname = outdir + 'ET' + os.path.basename(e_filename)[2:]
        write_hdf(h5_output, newname)
    if 'l' in ocomp:
        h5_output.data = np.sqrt(np.abs(np.square(e1data) + np.square(e2data)))
        h5_output.run_attributes['NAME'][0], h5_output.data_attributes['LONG_NAME'] = 'EL', 'E_L'
        newname = outdir + 'EL' + os.path.basename(e_filename)[2:]
        write_hdf(h5_output, newname)
    return e_filename


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
