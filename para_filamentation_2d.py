##################################################################################
# this script calculates growth of filamentation in LPI
# generated plasmas.  In English, it looks at time history as a function of 
# k_perp near k_parallel = 0.  
# the script processes the entire 
###################################################################################
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

# *****************************************************************************
# *****************************************************************************
# MPI related
import mpi4py
from mpi4py import MPI
# MPI related
# *****************************************************************************
# *****************************************************************************

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

def file_name_field(path,field,file_no):
    filename=path+'/FLD/'+field+'/'+field+'-'+repr(file_no).zfill(6)+'.h5'
    # print(filename)
    return(filename)
    
 
def file_name_phase(path,field,species,file_no):
    filename=path+'/PHA/'+field+'/'+species+'/'+field+'-'+species+'-'+repr(file_no).zfill(6)+'.h5'
    # print(filename)
    return(filename)

def file_name_density(path,field,species,file_no):
    filename=path+'/DENSITY/'+species+'/'+field+'/'+field+'-'+species+'-'+repr(file_no).zfill(6)+'.h5'
    # print(filename)
    return(filename)

def dir_name_density(path,field,species):
    dirname = path+'/DENSITY/'+species+'/'+field+'/'
    return(dirname)




def print_help():
    print('mpirun -n X python para_filamentation_2d.py <xmin> <xmax> <InputDir> <OutputName>')
    print('options:')
    print('  -s: only process every S files')

# *****************************************************************************
# *****************************************************************************
# MPI initialization
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
# *****************************************************************************
# *****************************************************************************
# *****************************************************************************


argc = len(sys.argv)
try:
    opts, args = getopt.gnu_getopt(sys.argv[1:], "hs:", ['avg', 'env'])
except getopt.GetoptError:
    print_help()
    sys.exit(2)

if len(args) < 4:
    print_help()
    sys.exit(2)
xmin = float(args[1])
xmax = float(args[2])
dirName = args[3]
outFilename = args[4]
dir_ext = ''
# default skip is 1
skip = 1    
#
# set defaults for xmin and xmax
xmin=300
xmax=400
#
tags = ''
for opt, arg in opts:
    if opt == '-h':
        print_help()
        sys.exit()
    elif opt == '-s':
        skip = arg
    else:
        print(print_help())
        sys.exit(2)

ion_species_name = 'species_2'

ion_files = sorted(glob.glob(dirName  + dir_ext + '/*.h5'))

total_time = len(ion_files)


my_share = total_time // size 
i_begin = rank * my_share
if rank < (size - 1):
    i_end = (rank + 1) * my_share
else:
    i_end = total_time


avg_array=np.ones(n_avg)/n_avg
#
# read the second file to get the time-step
#
h5_filename = ion_files[0]  
rho_0 = osh5io.read_h5(h5_filename)
rho_0_inverse = osh5utils.fft2(rho_0(rho_0.loc[:,xmin:xmax]))

array_dims = rho_0_inverse.shape
nx = array_dims[0]
ny = array_dims[1]

h5_data = osh5io.read_h5(ion_files[1])
time_step = h5_data.run_attrs['TIME'][0]
# h5_output = hdf_data()
# h5_output.shape = [total_time, nx]
print('nx=' + repr(nx))
print('ny=' + repr(ny))
print('time_step=' + repr(time_step))
print('total_time=' + repr(total_time))

# Here we initialize 2 variables
h5_output = np.zeros((total_time, nx))
total = np.zeros((total_time,nx))

#total = 0
total2 = 0
rho_0 = osh5io.read_h5()

kxaxis = rho_0_inverse.axes[0]

taxis=osh5def.DataAxis(0, time_step * (total_time -1), total_time,
    attrs={'NAME':'t', 'LONG_NAME':'time', 'UNITS':'1 / \omega_p'})

data_attrs = { 'UNITS': osh5def.OSUnits('m_e \omega_p^3'), 'NAME': 's1', 'LONG_NAME': 'S_1' }

print(repr(kxaxis.min))
print(repr(kxaxis.max))
run_attrs = {'XMAX' : np.array( [time_step * (total_time-1), kxaxis.max] ) , 
            'XMIN' : np.array( [0, kxaxis.min ] ) }

# h5_output.run_attrs['TIME'] = 0.0
# h5_output.run_attrs['UNITS'] = 'm_e /T'



file_number = 0
for file_number in range(i_begin, i_end,skip):
    # print(file_number)
    ion_filename = ion_files[file_number]
    
    ion_data = osh5io.read_h5(ion_filename)
    del_ion = ion_data - rho_0
    #if(file_number % 10 == 0):
        #print(s1_data.shape)
    rho_inverse= np.abs(osh5utils.fft2(del_ion.loc[:,xmin:xmax]))
    # print(temp.shape)
    h5_output[file_number, 1:nx] = rho_inverse[:,0.0]


comm.Reduce(h5_output, total, op=MPI.SUM, root=0)
if rank == 0:
    h5_output = total

if rank == 0:
    b=osh5def.H5Data(h5_output, timestamp='x', data_attrs=data_attrs,run_attrs=run_attrs, axes=[taxis, xaxis])
    osh5io.write_h5(b,filename=outFilename)
