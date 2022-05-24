#
# extracting forward and backward components of e2.  suitable for 1D data
#
# FST, (c) 2019 Regents of The University of California
#

import sys
# This is the path for Frank's Mac Pro
#
sys.path.append('/Users/franktsung/Documents/codes/python-tsung/')
sys.path.append('/Volumes/Lacie-5TB/codes/pyVisOS/')
#
#

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
from mpi4py import MPI


def print_help():
    print('python para_hfhi_growth.py [options] <zmin> <zmax> <kmax> <InputDir> <OutputDir>')
    print('zmin = minimum z value saved')
    print('zmax = maximum z value saved')
    print('kmax = maximum k_perp saved')
    print('InputDir - Location of the MS folder')
    print('OutputDir - Location of the output folder')
    print('options:')
    print(' None Implemented Yet')

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
zmin = args[0]
zmax = args[1]
kmax = args[2]
dirName = args[3]
outDir = args[4]
dir_ext = ''

for opt, arg in opts:
    if opt == '-h':
        print_help()
        sys.exit()
    elif opt == '--avg':
        dir_ext = '-savg'
    elif opt == '--env':
        dir_ext = '-senv'
    elif opt == '-n':
        density = float(arg)
    else:
        print(print_help())
        sys.exit(2)

index_of_refraction = np.sqrt(1-density)

v_phase = 1/index_of_refraction

print( repr(index_of_refraction)+' , '+repr(v_phase) )




e1 = sorted(glob.glob(dirName + '/FLD/e1' + dir_ext + '/*.h5'))

e2 = sorted(glob.glob(dirName + '/FLD/e2' + dir_ext + '/*.h5'))


total_time = len(e1)
my_share = total_time // size
i_begin = rank * my_share
if rank < (size - 1):
    i_end = (rank + 1) * my_share
else:
    i_end = total_time
part = total_time / size

#
# read the second file to get the time-step
#
h5_filename = e1[1]  
h5_data = osh5io.read_h5(h5_filename)
array_dims = h5_data.shape

# 
# the array is transposed due to Fortran array ordering
nx = array_dims[0]
ny = array_dims[1]
#
#


time_step = h5_data.run_attrs['TIME'][0]
# h5_output = hdf_data()
# h5_output.shape = [total_time, nx]
print('nx=' + repr(nx))
print('ny=' + repr(ny))
print('time_step=' + repr(time_step))
print('total_time=' + repr(total_time))



# calculate the z-range
dz=h5_data.axis[1][1]-h5_data.axis[1][0]
nx_begin=int((zmin-h5_data.axis[1][0])/dz)
nx_end=int((zmax-h5_data.axis[1][0])/dz)
xaxis=h5_data.axes[1][nx_begin:nx_end]
#
taxis=osh5def.DataAxis(0, time_step * (total_time -1), total_time,
    attrs={'NAME':'t', 'LONG_NAME':'time', 'UNITS':'1 / \omega_p'})
delta_k = 2*np.pi/(h5_data.axis[0].max-h5_data.axis[0].min)
nk = int(kmax/delta_k)

nz=nx_end-nx_begin+1


# here we allocate the history array
total_es = np.zeros((total_time,nk,nz))
total_em =np.zeros((total_time,nk,nz))

total_es = 0
total_em = 0


kaxis=osh5def.DataAxis(0, nk* delta_k, nk, attrs{'NAME':'k_{\perp}', 'LONG_NAME':'k_{\perp}', 'UNITS':osh5def.OSUnits('\omega_{0}/c')})
data_attrs_hfhi_es_hist = { 'UNITS': osh5def.OSUnits('a.u.'), 'NAME': 'ES modes history', 'LONG_NAME': 'hfhi_es_hist' }

data_attrs_hfhi_em_hist = { 'UNITS': osh5def.OSUnits('a.u.'), 'NAME': 'EM modes history', 'LONG_NAME': 'hfhi_em_hist' }


run_attrs = {'XMAX' : np.array( [ time_step * (total_time-1),kmax,zmax] ) , 
            'XMIN' : np.array( [0, 0, zmin] ) }



i_count = 0
file_number = 0
for file_number in range(i_begin, i_end):
    e1_filename = e1[file_number]
    e2_filename = e2[file_number]
    
    if (i_count % 10 == 0 and rank == 0): 
        print(e1_filename)
    i_count = i_count+1
    e1_data = osh5io.read_h5(e1_filename)
    e2_data = osh5io.read_h5(e2_filename)

    el,et = field_decompose((e1_data,e2_data),outquants=(l1,t2))
    el_inv = np.abs(np.fft.fft(el,axis=0))
    et_inv = np.abs(np.fft.fft(et,axis=0))
    
    # temp = np.abs(np.fft.fft(e2_data,axis=0))
    total_es[file_number,:,:]=temp[0:nk,nx_begin:nx_end]
    total_em[file_number,:,:]=temp[0:nk,nx_begin:nx_end]

#    temp = np.sum(s1_data, axis=0) / nx


# sum up the results to node 0 and output, 2 datasets, one for +
# component abd one for the - component
# first let's do the + root
comm.Reduce(e2_plus_output, total, op=MPI.SUM, root=0)
if rank == 0:
    b=osh5def.H5Data(total, timestamp='x', data_attrs=data_attrs_eplus,
        run_attrs=run_attrs, axes=[taxis,xaxis])
    outFilename=outDir+'/'+'hfhi-es-data.h5'
    osh5io.write_h5(b,filename=outFilename)

# now let's do the - component
comm.Reduce(total2, total, op=MPI.SUM, root=0)
if rank == 0:
    b=osh5def.H5Data(total2, timestamp='x', data_attrs=data_attrs_eminus,
                     run_attrs=run_attrs, axes=[taxis,xaxis])
    outFilename=outDir+'/'+'e2-minus.h5'
    osh5io.write_h5(b,filename=outFilename)
#    write_hdf(h5_output, outFilename)
print('Before barrier'+repr(rank))
comm.barrier()

