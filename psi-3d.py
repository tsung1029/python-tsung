from hdf_adam import *

for filenumber in range(46,105,1):
	strnum = str(filenumber)
	strlen = len(strnum)
	for zero in range(strlen,6):
		 strnum = '0' + strnum
	filename_in = '/u/d2scratch/tsung/dla-3d/llnl-ring/e1-savg/e1-savg-'+strnum+'.h5'
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
        temp[:,:,nx3-1] = 0.0
        for i in range(nx3-2,1,-1): 
		#temp[:,:,i] = temp[:,:,i+1]+0.8*(data.data[:,:,i]+data.data[:,:,i-1])
                # print i
		# temp[150,150,i] = temp[150,150,i+1]+1.6*data.data[150,150,i+1]
		#temp[i,:,:] = temp[i+1,:,:]+0.8*(data.data[i,:,:]+0.5*data.data[i-1,:,:]+0.5*data.data[i-2,:,:])
		# temp[:,:,i] = temp[:,:,i+1]+0.8*(data.data[:,:,i+1]+0.5*data.data[:,:,i]+0.5*data.data[:,:,i-1])
                temp[:,:,i] = temp[:,:,i+1]+0.8*(data.data[:,:,i]+0.5*data.data[:,:,i-1]+0.5*data.data[:,:,i-2])
        data.data=temp
	print data.data_attributes['LONG_NAME']
	data.data_attributes['LONG_NAME'] = '\psi'
	filename_out = '/u/d2scratch/tsung/dla-3d/llnl-ring/psi/psi'+strnum+'.h5'
	#filename_out = '/Users/mzoufras/Desktop/PHA2/x2x1/x2x1-'+strnum+'.h5'
	print 'Writing ' + filename_out
 	write_hdf(data,filename_out,dataset_name ='PSI')
        data.slice(x3=150)
	data.data_attributes['LONG_NAME'] = '\psi_{slice}'
	filename_out = '/u/d2scratch/tsung/dla-3d/llnl-ring/psi-slice/psi'+strnum+'.h5'
	write_hdf(data,filename_out)
