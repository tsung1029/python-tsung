#
# *******************************************************************
# *******************************************************************
# new python 3.X routines for OSIRIS data analysis
#
# *******************************************************************
# *******************************************************************

import sys
sys.path.append('/Users/franktsung/Documents/codes/python-tsung/')
sys.path.append('/Volumes/Lacie-5TB/codes/pyVisOS/')

# *******************************************************************
# *******************************************************************

# import h5_utilities

# *******************************************************************
# *******************************************************************
%matplotlib notebook
#%matplotlib widget
import numpy as np
import sys
# *******************************************************************
# *******************************************************************

# *******************************************************************
# *******************************************************************
import osh5io
import osh5def
import osh5vis
import osh5utils
import osh5visipy
import matplotlib.pyplot as plt
import hfhi
# *******************************************************************
# *******************************************************************

# *******************************************************************
# *******************************************************************
import numpy
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
# *******************************************************************
# *******************************************************************

def phase_space_average_new(path,prefix,fileno,interval,nfiles):
    file_begin=fileno
    file_end=fileno+((nfiles)*interval)
    filename=path+prefix+repr(file_begin).zfill(6)+'.h5'
    # print(filename)
    hdf5_data=osh5io.read_h5(filename)
    # print(hdf5_data)
    xmin=hdf5_data.axes[0].min
    xmax=hdf5_data.axes[0].max

    ymin=hdf5_data.axes[1].min
    ymax=hdf5_data.axes[1].max

# dx=(xmax-xmin)/float(nx[0]-1)
# dy=(ymax-ymin)/float(nx[1]-1)
# print(dx)
# print(dy)
# xmax=xmax-dx
# ymax=ymax-dy
# xmin=xmin+10.0*dx
# ymin=ymin-10.0*dy
# print(repr(xmin)+' '+repr(xmax)+' '+repr(ymin)+' '+repr(ymax))
    nx=hdf5_data.shape
    xaxis=numpy.linspace(xmin,xmax,num=nx[0])
    yaxis=numpy.linspace(ymin,ymax,num=nx[1])
    
    total_files=1
    # print(repr(file_begin)+' '+repr(file_end)+' '+repr(nfiles))
    for i_file in range(file_begin+interval,file_end,interval):
        # print(repr(i_file))
        filename=path+prefix+repr(i_file).zfill(6)+'.h5'
        # print(filename)
        try:
            hdf5_data_temp=osh5io.read_h5(filename)
            total_files=total_files+1    
            xmin_temp=hdf5_data_temp.axes[0].min
            xmax_temp=hdf5_data_temp.axes[0].max
            ymin_temp=hdf5_data_temp.axes[1].min
            ymax_temp=hdf5_data_temp.axes[1].max
            nx_temp=hdf5_data_temp.shape
            if((xmin==xmin_temp) and (xmax==xmax_temp) and (ymin==ymin_temp) and (ymax==ymax_temp)):
                # print('passed')
                hdf5_data=numpy.abs(hdf5_data)+numpy.abs(hdf5_data_temp)
            else:
                # print('no pass')
                xaxis_temp=numpy.linspace(xmin_temp,xmax_temp,nx_temp[0])
                yaxis_temp=numpy.linspace(ymin_temp,ymax_temp,nx_temp[1])
                interp_func=interpolate.interp2d(xaxis_temp,yaxis_temp,hdf5_data_temp,kind='cubic')
                temp_data=numpy.zeros((nx[0],nx[1]))
                for ix in range(0,nx[0]):
                    for iy in range(0,nx[1]):
                        temp_data[iy,ix]=interp_func(xaxis[ix],yaxis[iy])
        
                hdf5_data=numpy.abs(hdf5_data)+numpy.abs(temp_data)
        except:
            print('error')
            continue
        

    # print(repr(total_files))   
    hdf5_data=hdf5_data/float(total_files)
    return hdf5_data



# *******************************************************************
# *******************************************************************
def srs_quick_look(path,fileno):
	path_e1=path_main+'/MS/FLD/e1/'
	file_prefix_e1='e1-'

	path_e2=path_main+'/MS/FLD/e2/'
	file_prefix_e2='e2-'

	path_e3=path_main+'/MS/FLD/e3/'
	file_prefix_e3='e3-'

	path_b1=path_main+'/MS/FLD/b1/'
	file_prefix_b1='b1-'

	path_b2=path_main+'/MS/FLD/b2/'
	file_prefix_b2='b2-'

	path_b3=path_main+'/MS/FLD/b3/'
	file_prefix_b3='b3-'


	path_p1x1_e = path_main+'/MS/PHA/p1x1/species_1/'
	file_prefix_p1x1_e = 'p1x1-species_1-'
	# fileno = 22500
	interval = 30
	nfiles=10

	path_p1p2_e = path_main+'/MS/PHA/p1p2/species_1/'
	file_prefix_p1p2_e = 'p1p2-species_1-'


	from scipy import signal




	filename_e1=path_e1+file_prefix_e1+repr(fileno).zfill(6)+'.h5'
	filename_e2=path_e2+file_prefix_e2+repr(fileno).zfill(6)+'.h5'
	filename_e3=path_e3+file_prefix_e3+repr(fileno).zfill(6)+'.h5'
	filename_b1=path_b1+file_prefix_b1+repr(fileno).zfill(6)+'.h5'
	filename_b2=path_b2+file_prefix_b2+repr(fileno).zfill(6)+'.h5'
	filename_b3=path_b3+file_prefix_b3+repr(fileno).zfill(6)+'.h5'

	filename_p1x1_e = path_p1x1_e+file_prefix_p1x1_e+repr(fileno).zfill(6)+'.h5'
	filename_p1p2_e = path_p1p2_e+file_prefix_p1p2_e+repr(fileno).zfill(6)+'.h5'


	# print(filename)
	e1 = osh5io.read_h5(filename_e1)
	e2 = osh5io.read_h5(filename_e2)
	e3 = osh5io.read_h5(filename_e3)

	b1 = osh5io.read_h5(filename_b1)
	b2 = osh5io.read_h5(filename_b2)
	b3 = osh5io.read_h5(filename_b3)


	# print(e1.axes[0].attrs)
	# Method 1:  Just read files
	# p1x1_e = osh5io.read_h5(filename_p1x1_e)
	# p1p2_e = osh5io.read_h5(filename_p1p2_e)

	# Method 2:  Average over N steps
	interval=30
	nfiles = 20
	p1x1_e = phase_space_average_new(path_p1x1_e,file_prefix_p1x1_e,fileno,interval,nfiles)
	p1p2_e = phase_space_average_new(path_p1p2_e,file_prefix_p1p2_e,fileno,interval,nfiles)


	SMALL_SIZE = 12
	MEDIUM_SIZE = 14
	BIGGER_SIZE = 18

	plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
	plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
	plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
	plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
	plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
	plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
	plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


	s1 = e2*b3 - e3 * b2
	s1.data_attrs['NAME']='S1'
	s1.data_attrs['LONG_NAME']='S_1'

	n_smooth=200
	smoothing=np.ones((1,n_smooth))/n_smooth

	n_avg = 200
	smooth_array=np.ones((1,n_avg))/n_avg
	temp=signal.convolve2d(s1,smooth_array,mode='same',boundary='wrap')
	s1[:,:]=temp[:,:]
	s1 = np.average(s1,axis=0)

	plt.figure(figsize=(10,7.0))


	plt.subplot(222)
	temp=signal.convolve2d(e1,smoothing,mode='same',boundary='wrap')
	osh5vis.osplot(np.average(e1,axis=0))
	plt.ylim(-0.02,0.02)

	# plt.subplot(222)
	# osh5vis.osplot(e2,cmap='seismic',vmin=-0.02,vmax=0.02)

	# plt.subplot(223)
	# osh5vis.osplot(s1,cmap='PuBu',vmin=0,vmax=0.000025)

	plt.subplot(221)
	p1x1_e = numpy.abs(p1x1_e)
	data_max= numpy.amax(p1x1_e)
	p1x1_e=p1x1_e/data_max
	osh5vis.oscontourf(p1x1_e,levels=[10**-5,3.0*10**-3,5*10**-3,1*10**-2,10**-1,3*10**-1,5*10**1],norm=LogNorm(),cmap='terrain',vmin=1e-3,vmax=100)
	osh5vis.oscontour(p1x1_e,levels=[10**-5,3.0*10**-3,5*10**-3,1*10**-2,10**-1,3*10**-1,5*10**1],norm=LogNorm(),colors='black',linestyles='dashed',vmin=1e-5,vmax=500,colorbar=False)

	plt.subplot(223)
	# osh5vis.osplot(np.log(np.abs(p1p2_e)+1e-5),cmap='hsv')

	p1p2_e = numpy.abs(p1p2_e)
	data_max= numpy.amax(p1p2_e)
	p1p2_e=p1p2_e/data_max
	osh5vis.oscontourf(p1p2_e,levels=[10**-6,3.0*10**-3,5*10**-3,1*10**-2,10**-1,3*10**-1,5*10**1],norm=LogNorm(),cmap='terrain',vmin=1e-3,vmax=100)
	osh5vis.oscontour(p1p2_e,levels=[10**-6,3.0*10**-3,5*10**-3,1*10**-2,10**-1,3*10**-1,5*10**1],norm=LogNorm(),colors='black',linestyles='dashed',vmin=1e-5,vmax=500,colorbar=False)
	# plt.xlim(-0.8,0.8)
	plt.xlim(-0.35,0.35)
	plt.subplot(224)
	osh5vis.osplot(s1, title='Axial <Poynting Flux>')
	plt.ylim(0,0.00009)
	plt.tight_layout()
	plt.show()

# *******************************************************************
# *******************************************************************


# *******************************************************************
# *******************************************************************
def upic_bvp_2(rundir,*args,**kwpassthrough):
#2345
    import os


    def something_2(rundir,file_no):
        v_phase = 1/.3
        v_group = 3/v_phase
        my_path=os.getcwd()
        #print(my_path)
        working_dir=my_path+'/'+rundir
        #print(working_dir)
        
        
        dir_0=rundir+'/Vx_x/'
        dir_1=rundir+'/Ex/'
        
        
        

        quan_0_prefix='vx_x_'
        quan_1_prefix='Ex-0_'
        
        
        fig = plt.figure(figsize=(10,8) )

        
        phase_space_filename=dir_0+quan_0_prefix+repr(file_no).zfill(6)+'.h5'
        e1_filename=dir_1+quan_1_prefix+repr(file_no).zfill(6)+'.h5'
        
        
        phase_space_data = osh5io.read_h5(phase_space_filename)
        e1_data = np.average(osh5io.read_h5(e1_filename),axis=0)
        fig.suptitle('Landau Boundary Problem, time = '+repr(phase_space_data.run_attrs['TIME'][0])+' $\omega_p^{-1}$ \n \n')

        p1_x1_plot = plt.subplot(221)
        osh5vis.osplot(np.log(np.abs(phase_space_data)+1e-10),title='Phase Space',vmax=10,vmin=-2,cmap='Blues')
        # =====================================================================
        # =====================================================================
        # the following lines draw vertical axis indicating the group velocity
        # p1_x1_plot.axvline(x=512-v_group*phase_space_data.run_attrs['TIME'][0])
        # p1_x1_plot.axvline(x=512+v_group*phase_space_data.run_attrs['TIME'][0])
        # =====================================================================
        # =====================================================================
        e1_plot = plt.subplot(222)
        osh5vis.osplot(e1_data,title='Electric Field')
        # plt.tight_layout()
        fig.subplots_adjust(top=0.82)
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
        
        fig.subplots_adjust(top=0.82)
        
        fv_slices = plt.subplot(224)
        
        
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
        plt.plot(xaxis,np.average(phase_space_data[:,492:532],axis=1))
        plt.plot(xaxis,np.average(phase_space_data[:,352:372],axis=1),'r')
        plt.plot(xaxis,np.average(phase_space_data[:,652:672],axis=1),'g')
        plt.xlabel('$v/v_{th}$')
        plt.ylabel('f(v) [a.u]')
        fv_slices.set_ylim([0, 1500])
        
        
        plt.tight_layout()
        plt.show()
        # plt.show()
        
#2345
    # my_path=os.getcwd()
    # working_dir=my_path+'/'+rundir
    phase_space_dir=rundir+'/Vx_x/'
    files=sorted(os.listdir(phase_space_dir))
    print(files[1])
    start=files[1].find('vx_x_')+5
    end=files[1].find('.')
    print(files[1][start:end])
    file_interval=int(files[1][start:end])
    file_max=(len(files)-1)*file_interval
    print(file_max)

    interact(something_2,rundir=rundir,file_no=widgets.IntSlider(min=0,max=file_max,step=file_interval,value=0), continuous_update=False)
    #something(rundir=rundir,file_no=20)

# *******************************************************************
# *******************************************************************
