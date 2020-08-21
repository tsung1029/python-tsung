#
# ************************************************************
# ************************************************************
# This subroutine calculates poynting flux in the q3d geometry
# ************************************************************
# ************************************************************
#

import sys

# including my path and Han Wen's path
sys.path.append('/Users/franktsung/Documents/codes/python-tsung/')
sys.path.append('/Volumes/Lacie-5TB/codes/pyVisOS/')
#


import osh5io
import osh5def
import osh5vis
import osh5utils

from h5_utilities import *
import matplotlib.pyplot as plt
import sys
# command line options (argc, argv stuff)
import getopt 
#

# glob -> finding files in a directory
import glob
#

import numpy as np
from mpi4py import MPI

# ************************************************************
# ********              utility routines            **********
# ************************************************************

def filename_re(rundir,plasma_field,mode,fileno):
    plasma_field_path=rundir+'/FLD/MODE-{mode:1d}-RE/{plasma_field:s}_cyl_m/{plasma_field:s}_cyl_m-{mode:1d}-re-{fileno:06d}.h5'.format(plasma_field=plasma_field,mode=mode,fileno=fileno)
    # print(plasma_field_path)
    return(plasma_field_path)

def filename_im(rundir,plasma_field,mode,fileno):
    plasma_field_path=rundir+'/FLD/MODE-{mode:1d}-IM/{plasma_field:s}_cyl_m/{plasma_field:s}_cyl_m-{mode:1d}-im-{fileno:06d}.h5'.format(plasma_field=plasma_field,mode=mode,fileno=fileno)
    # print(plasma_field_path)
    return(plasma_field_path)

def dirname_re(rundir,plasma_field,mode):
    plasma_field_path=rundir+'/FLD/MODE-{mode:1d}-RE/{plasma_field:s}_cyl_m'.format(plasma_field=plasma_field,mode=mode)
    # print(plasma_field_path)
    return(plasma_field_path)

def dirname_im(rundir,plasma_field,mode):
    plasma_field_path=rundir+'/FLD/MODE-{mode:1d}-IM/{plasma_field:s}_cyl_m'.format(plasma_field=plasma_field,mode=mode)
    # print(plasma_field_path)
    return(plasma_field_path)


def print_help():
    print('para_poynt_q3d.py [options] <InputDir> <OutputName>')
    print('options:')
    print('  -n: average over n grid points at entrance (32 by default)')
    print('  -m: number of modes to include')
    print('  --avg: look into the -savg directory')
    print('  --env: look into the -senv directory')
# ************************************************************
# ************************************************************
# ************************************************************

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
n_modes = 1
#
# sumdir = 0 summing over the transverse direction
sumdir = 0
#
#
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
    elif opt == '-m':
        n_modes = arg
    else:
        print(print_help())
        sys.exit(2)

# e2 = sorted(glob.glob(dirname_re(dirName,'e2',0) + dir_ext + '/*.h5'))
# e3 = sorted(glob.glob(dirName + '/MS/FLD/e3' + dir_ext + '/*.h5'))
e2 = sorted(glob.glob(dirName + '/MS/FLD/b2' + dir_ext + '/*.h5'))
# b3 = sorted(glob.glob(dirName + '/MS/FLD/b3' + dir_ext + '/*.h5'))
start=e2[1].find('e2_cyl_m-0-re-')+15
end=e2[1].find('.')
i_begin=int(e2[0][start:end])
file_interval=int(e2[1][start:end])

total_time = len(e2)
i_end=i_begin+total_time*file_interval
my_share = total_time // size
i_begin = rank * my_share
if rank < (size - 1):
    i_end = (rank + 1) * my_share
else:
    i_end = total_time
part = total_time / size
avg_array=np.ones(n_avg)/n_avg
#
# read the second file to get the time-step
#
# 
h5_filename = e2[1]  
h5_data = osh5io.read_h5(h5_filename)
array_dims = h5_data.shape
nx = array_dims[0]
dx=h5_data.axes[0]
ny = array_dims[1]

time_step = h5_data.run_attrs['TIME'][0]
# h5_output = hdf_data()
# h5_output.shape = [total_time, nx]
print('nx=' + repr(nx))
print('ny=' + repr(ny))
print('time_step=' + repr(time_step))
print('total_time=' + repr(total_time))
print('i_begin =' + repr(i_begin))
print('i_end =' + repr(i_end))
print('file_interval =' + repr(file_interval))

h5_output = np.zeros((total_time, ny))
total = np.zeros((total_time,ny))

#total = 0
total2 = 0
# if rank == 0:
#    total = np.zeros((total_time, nx))

raxis=h5_data.axes[0]
dr=(raxis.max-raxis.min)/ny
xaxis=h5_data.axes[1]
taxis=osh5def.DataAxis(0, time_step * (total_time -1), total_time,
    attrs={'NAME':'t', 'LONG_NAME':'time', 'UNITS':'1 / \omega_p'})

data_attrs = { 'UNITS': osh5def.OSUnits('m_e \omega_p^3'), 'NAME': 'Power', 'LONG_NAME': 'Trans. Int. Power' }

print(repr(xaxis.min))
print(repr(xaxis.max))
run_attrs = {'XMAX' : np.array( [time_step * (total_time-1), xaxis.max] ) , 
            'XMIN' : np.array( [0, xaxis.min ] ) }

# h5_output.run_attrs['TIME'] = 0.0
# h5_output.run_attrs['UNITS'] = 'm_e /T'



file_number = 0
skip = 1
temp=np.zeros(ny)
for file_number in range(i_begin, i_end):
    file_index=i_begin+file_number*file_interval
    for mode_number in range(0,n_modes+1):

        e2_filename_re = filename_re(dirName,'e2',mode_number,file_index)
        e3_filename_re = filename_re(dirName,'e3',mode_number,file_index)
        b2_filename_re = filename_re(dirName,'b2',mode_number,file_index)
        b3_filename_re = filename_re(dirName,'b3',mode_number,file_index)
        e2_filename_im = filename_im(dirName,'e2',mode_number,file_index)
        e3_filename_im = filename_im(dirName,'e3',mode_number,file_index)
        b2_filename_im = filename_im(dirName,'b2',mode_number,file_index)
        b3_filename_im = filename_im(dirName,'b3',mode_number,file_index)
        if( rank == 0 and file_number % 10 == 0):
            print(e2_filename_re)
        e2_data_re = osh5io.read_h5(e2_filename_re)
        e3_data_re = osh5io.read_h5(e3_filename_re)
        b2_data_re = osh5io.read_h5(b2_filename_re)
        b3_data_re = osh5io.read_h5(b3_filename_re)
        if (mode_number!=0):
            e2_data_im = osh5io.read_h5(e2_filename_im)
            e3_data_im = osh5io.read_h5(e3_filename_im)
            b2_data_im = osh5io.read_h5(b2_filename_im)
            b3_data_im = osh5io.read_h5(b3_filename_im)
        if (mode_number == 0):
            s1_data = e2_data_re * b3_data_re - e3_data_re * b2_data_re
        else:
            s1_data = s1_data + 0.5 * (e2_data_re * b3_data_re - e3_data_re * b2_data_re)
            + 0.5 * (e2_data_im * b3_data_im - e3_data_im * b2_data_im)
    #if(file_number % 10 == 0):
        #print(s1_data.shape)
    # instead of the sum now integrates it as int (r * S(r))
    #
    print(s1_data.shape)
    print(dr)
    int_const = 2*3.1415926*dr*dr
    for i_z in range(0,ny):
        temp[i_z] = 0.0
        for i_r in range(0,nx):
            temp[i_z]=temp[i_z]+int_const*(i_r+0.5)*s1_data[i_r,i_z]


    # temp=np.sum(s1_data,axis=0) / nx
    print(temp.shape)
    h5_output[file_number, 1:ny] = np.abs(np.convolve(temp[1:ny],avg_array,mode='same'))
    # h5_output[file_number, 1:nx] = np.convolve(s1_data[1:nx],avg_array,mode='same')
#    temp = np.sum(s1_data, axis=0) / nx
#    h5_output.data[file_number, 1:ny] = temp[1:ny]
#    temp = np.sum(s1_data[0:n_avg, :], axis=0) / n_avg
#    income[file_number, 1:ny] = temp[1:ny]
#    temp = np.sum(s1_data, axis=1) / ny
#    h5_output2.data[file_number, 1:nx] = temp[1:nx]
#    temp = np.sum(s1_data[:, 0:n_avg], axis=1) / n_avg
#    income2[file_number, 1:nx] = temp[1:nx]
    # file_number+=1

comm.Reduce(h5_output, total, op=MPI.SUM, root=0)
if rank == 0:
    b=osh5def.H5Data(total, timestamp='x', data_attrs=data_attrs, 
        run_attrs=run_attrs, axes=[taxis, xaxis])
    osh5io.write_h5(b,filename=outFilename)
#    write_hdf(h5_output, outFilename)
print('Before barrier'+repr(rank))
comm.barrier()
# comm.Reduce(income, total, op=MPI.SUM, root=0)
# if rank == 0:
#     h5_output.data = total
#     newName = outFilename.rsplit('.', 1)[0] + '-x-n' + str(n_avg) + '.h5'
#     write_hdf(h5_output, newName)
# comm.barrier()
# comm.Reduce(h5_output2.data, total2, op=MPI.SUM, root=0)
# if rank == 0:
#     h5_output2.data = total2
#     write_hdf(h5_output2, outFilename)
# comm.barrier()
# comm.Reduce(income2, total2, op=MPI.SUM, root=0)
# if rank == 0:
#     h5_output2.data = total2
#     newName = outFilename.rsplit('.', 1)[0] + '-y-n' + str(n_avg) + '.h5'
#     write_hdf(h5_output2, newName)
