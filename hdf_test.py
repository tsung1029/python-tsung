import numpy
import hdf_adam
import matplotlib.pyplot as plt

test=hdf_adam.hdf_data()

nx=200
ny=100
test.data=numpy.zeros((nx,ny))
for i in range(0,nx):
  for j in range(0,ny):
    test.data[i,j]=numpy.cos(8.0*numpy.pi*float(i)/float(nx))*numpy.sin(12.0*numpy.pi*float(j)/float(ny))


test.run_attributes['TIME']=0.0
test.run_attributes['UNITS']='AU'
test.run_attributes['LONG_NAME']='My Data'
test.axes=[hdf_adam.data_basic_axis(0,0,numpy.pi*4.0,nx),hdf_adam.data_basic_axis(1,0,numpy.pi*2.0,ny)]
test.axes[0].attributes['LONG_NAME']='x'
test.axes[1].attributes['LONG_NAME']='y'
test.axes[0].attributes['UNITS']='AU'
test.axes[1].attributes['UNITS']='AU'
test.shape=[nx,ny]

hdf_adam.write_hdf(test,filename='test.h5')
