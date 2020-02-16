import numpy as np
# the rwigner function takes a 1D numpy array and makes a wigner transform. 
#
def rwigner(data):
    nx=data.shape[0]
    nxh = (nx+1)/2
    temp=np.zeros((nx,nx))
    wdata=np.zeros((nx,nxh))
    # temp1d=np.zeros(nx)
    dataplus=np.zeros(nx)
    dataminus=np.zeros(nx)
    temp[:,0]=data[:]
    
    for j in range(1,nx):
        dataplus=np.roll(data,j)
        dataminus=np.roll(data,-j)
        temp[:,j] = dataplus[:] * dataminus[:]
        
    
    for i in range(nx):
        temp1d=np.fft.fft(temp[:,i])
        wdata[i,:]=temp1d[0:nxh]
        
    return wdata
    
    
    
