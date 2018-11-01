from hdf_adam import *
from mpi4py import MPI


comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
print size,rank
starting = 46+rank
for filenumber in range(46+rank,105,size):
    strnum = str(filenumber)
    strlen = len(strnum)
    for zero in range(strlen,6):
        strnum = '0' + strnum
    filename_in = '/u/d2scratch/tsung/dla-3d/llnl-ring/e3-savg/e3-savg-'+strnum+'.h5'
    #filename_in = '/Volumes/Backup3D/BulBow3D_March2013/a5.3/MS/FLD/e1/e1-'+strnum+'.h5'
    print 'Reading ' + filename_in
    data = read_hdf(filename_in)
    #plotme(data)
    #show()
    temp=data.data
    nx1=data.axes[0].axis_numberpoints
    nx2=data.axes[1].axis_numberpoints
    nx3=data.axes[2].axis_numberpoints
    dx1=data.axes[0].increment
    dx2=data.axes[1].increment
    dx3=data.axes[2].increment
    print nx1,data.axes[0]
    print nx2,data.axes[1]
    print nx3,data.axes[2]
    print dx1,dx2,dx3
    temp[nx1-1,:,:] = 0.0
    for i in range(nx1-2,-1,-1):
    #temp[i,:,:] = temp[i+1,:,:]+dx1*data.data[i+1,:,:]
        temp[i,:,:] = temp[i+1,:,:]+1.6*data.data[i,:,:]
    data.data=temp
    dummy = np.zeros(1)
    if rank != 0 :
        comm.Recv(dummy,source=rank-1, tag=11)
    print data.data_attributes['LONG_NAME']
    data.data_attributes['LONG_NAME'] = 'a'
    filename_out = '/u/d2scratch/tsung/dla-3d/llnl-ring/a3/a3-'+strnum+'.h5'
    #filename_out = '/Users/mzoufras/Desktop/PHA2/x2x1/x2x1-'+strnum+'.h5'
    print 'Writing ' + filename_out
    write_hdf(data,filename_out,dataset_name ='a')
    data.slice(x3=150)
    data.data_attributes['LONG_NAME'] = 'a_{slice}'
    filename_out = '/u/d2scratch/tsung/dla-3d/llnl-ring/a3-slice/a3-'+strnum+'.h5'
    write_hdf(data,filename_out)
    if rank != (size-1):
        comm.Send(dummy,dest=rank+1,tag=11)
