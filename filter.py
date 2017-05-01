from hdf_adam import *
from scipy import signal
import scipy as sci

filename_in = 'original.h5'

data = read_hdf(filename_in)

filter_array=np.array([[[0,0,0],[0,0.166666,0],[0,0,0]],[[0,0.166666,0],[0.166666,1,0.166666],[0,0.166666,0]],[[0,0,0],[0,0.166666,0],[0,0,0]]])

fil_data = sci.signal.convolve(data.data,filter_array,mode='same')

# fil_data = sci.signal.wiener(data.data)

data.data = fil_data

filename_out = 'filtered.h5' 

write_hdf(data,filename_out)
