from hdf_adam import *
from mpi4py import MPI


#comm = MPI.COMM_WORLD
#size = comm.Get_size()
#rank = comm.Get_rank()
#print size,rank
#starting = 1+rank
for filenumber in range(1,105,1):
	strnum = str(filenumber)
	strlen = len(strnum)
	for zero in range(strlen,6):
		 strnum = '0' + strnum
	filename_in = '/u/d2scratch/tsung/llnl-none/MS/FLD/e2-savg/e2-savg-'+strnum+'.h5'
	#filename_in = '/Volumes/Backup3D/BulBow3D_March2013/a5.3/MS/FLD/e1/e1-'+strnum+'.h5'
	print 'Reading ' + filename_in
	data = read_hdf(filename_in)
	filename_in = '/u/d2scratch/tsung/llnl-none/MS/FLD/b3-savg/b3-savg-'+strnum+'.h5'
        data_2 = read_hdf(filename_in)
	#plotme(data)
        #data.data-=data_2.data
	#dummy = np.zeros(1)
	#if rank != 0 :
		#comm.Recv(dummy,source=rank-1, tag=11)
	print data.data_attributes['LONG_NAME']
	data.data_attributes['LONG_NAME'] = 'focusing force'
	filename_out = '/u/d2scratch/tsung/llnl-none/f2/f2-'+strnum+'.h5'
	#filename_out = '/Users/mzoufras/Desktop/PHA2/x2x1/x2x1-'+strnum+'.h5'
	print 'Writing ' + filename_out
 	write_hdf(data,filename_out,dataset_name ='F_2')
        data.slice(x3=150)
	data.data_attributes['LONG_NAME'] = 'F_{slice}'
	filename_out = '/u/d2scratch/tsung/llnl-none/f2-slice/f2-'+strnum+'.h5'
	write_hdf(data,filename_out)
	#if rank != (size-1):
		#comm.Send(dummy,dest=rank+1,tag=11)
