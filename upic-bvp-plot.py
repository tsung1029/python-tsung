# *******************************************************************************
# *******************************************************************************
#
# Making Movies from UPIC data (for spatial Landau Damping)
#
# FST, (c) 2020 Regents of The University of California
#
# *******************************************************************************
# *******************************************************************************

import sys
# change path depending on machine
# the path below is for Frank's Mac
#
sys.path.append('/Users/franktsung/Documents/codes/python-tsung/')
sys.path.append('/Volumes/Lacie-5TB/codes/pyVisOS/')

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
    print('python upic_bvp_movie.py [options] <InputDir> <OutputDir>')
    print('InputDir - Location of the MS folder')
    print('OutputDir - Location of the output folder, makes .png files from data')
    print('options:')
   
#2345
import os

# *******************************************************************************
# this is the subroutine pulled from the Jupyter notebook, replace the plotting
# with file-save
# *******************************************************************************
def something_2(rundir,file_no):

    my_path=os.getcwd()
    #print(my_path)
    working_dir=my_path+'/'+rundir
    #print(working_dir)
        
        
    dir_0=rundir+'/Vx_x/'
    dir_1=rundir+'/Ex/'
    
    output_dir = rundir +'/movies/'    
        
        

    quan_0_prefix='vx_x_'
    quan_1_prefix='Ex-0_'
    
    output_prefix='frame_'
        
        
    fig = plt.figure(figsize=(10,8) )

        
    phase_space_filename=dir_0+quan_0_prefix+repr(file_no).zfill(6)+'.h5'
    e1_filename=dir_1+quan_1_prefix+repr(file_no).zfill(6)+'.h5'
    
    output_filename=output_dir+'/'+output_prefix+repr(file_no).zfill(6)+'.png'
        
        
    phase_space_data = osh5io.read_h5(phase_space_filename)
    e1_data = np.average(osh5io.read_h5(e1_filename),axis=0)
        
    p1_x1_plot = plt.subplot(221)
    osh5vis.osplot(np.log(np.abs(phase_space_data)+1e-10),title='Phase Space',vmax=10,vmin=-2,cmap='Blues')
        
    e1_plot = plt.subplot(222)
    osh5vis.osplot(e1_data,title='Electric Field')
    e1_plot.set_ylim([-0.1,0.1])
    plt.tight_layout()
    fv_plot = plt.subplot(223)
        
        
    # print(hdf5_data)
    xmin=phase_space_data.axes[0].min
    xmax=phase_space_data.axes[0].max

    ymin=phase_space_data.axes[1].min
    ymax=phase_space_data.axes[1].max

# dx=(xmax-xmin)/float(nx[0]-1)
# dy=(ymax-ymin)/float(nx[1]-1)
# print(dx)
# print(dy)
# xmax=xmax-dx
# ymax=ymax-dy
# xmin=xmin+10.0*dx
# ymin=ymin-10.0*dy
# print(repr(xmin)+' '+repr(xmax)+' '+repr(ymin)+' '+repr(ymax))
    nx=phase_space_data.shape
    xaxis=np.linspace(xmin,xmax,num=nx[0])
    yaxis=np.linspace(ymin,ymax,num=nx[1])
    plt.plot(xaxis,np.average(phase_space_data,axis=1))
    plt.xlabel('$v/v_{th}$')
    plt.ylabel('f(v) [a.u]')
    fv_plot.set_ylim([0, 1500])
    plt.savefig(output_filename)
    # plt.show()
         


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
argc = len(sys.argv)
try:
    opts, args = getopt.gnu_getopt(sys.argv[1:], "hxyn:", ['avg', 'env'])
except getopt.GetoptError:
    print_help()
    sys.exit(2)


if len(args) < 1:
    print_help()
    sys.exit(2)
dirName = args[0]
# outDir = args[1]
dir_ext = ''
density = 0.1


#2345
    # my_path=os.getcwd()
    # working_dir=my_path+'/'+rundir
phase_space_dir=dirName+'/Vx_x/'
files=sorted(os.listdir(phase_space_dir))
print(files[1])
start=files[1].find('vx_x_')+5
end=files[1].find('.')
print('file interval = '+files[1][start:end])
## // --> floor division

file_interval=int(files[1][start:end])
file_max=(len(files)-1)*file_interval
total_time = len(files)

my_share = total_time // size
i_begin = rank * my_share
if rank < (size -1):
    i_end = (rank + 1) * my_share
else:
    i_end = total_time
    
print(file_max)

# *******************************************************************************
# this line replaces the slider
for file_number in range(i_begin, i_end):
    print('file_number = '+repr(file_number))
    something_2(rundir=dirName,file_no=file_number*file_interval)
# *******************************************************************************

