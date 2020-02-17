import numpy as np
# the rwigner function takes a 1D numpy array and makes a wigner transform. 
#

import osh5io
import osh5def
import osh5vis
import osh5utils


# ******************************************************************
from h5_utilities import *
import matplotlib.pyplot as plt
import sys
# Han Wen's pyVisOS
sys.path.append('/Volumes/Lacie-5TB/codes/pyVisOS/')


def os_rwigner(real_array,xaxis):

	nx=real_array.shape
	nxh=(nx+1)/2
    dx=(xaxis.max-xaxis.min)/nx
    kmax=(np.pi)/dx
    dk=(2.0*np.pi)/xaxis.max
	w_data= rwigner(real_array)

	kaxis=osh5def.DataAxis(0,kmax,nx_h,
		attrs={'NAME':'k','LONG_NAME':'wave number', 'UNITS': '\omega_0/c'}

	data_attrs = data_attrs = { 'UNITS': osh5def.OSUnits('[a.u.]'), 
	'NAME': 'W_{\phi}', 'LONG_NAME': 'Wigner Transform' }
	run_attrs = {'XMAX' : np.array( [kmax, xaxis.max] ) , 
            'XMIN' : np.array( [0, xaxis.min ] ) }

	os5data_wigner=osh5def.H5Data(w_data,timestamp='x', data_attrs=data_attrs, axes=[kaxis,xaxis])

	
