from hdf_adam import *

for filenumber in range(50,60,5):
    strnum = str(filenumber)
    strlen = len(strnum)
    for zero in range(strlen,6):
         strnum = '0' + strnum
    filename_in = '/Users/mzoufras/Desktop/noprepulse/x3x2x1/x3x2x1-s1-'+strnum+'.h5'
    #filename_in = '/Volumes/Backup3D/BulBow3D_March2013/a5.3/MS/PHA/x3x2x1/x3x2x1-s1-'+strnum+'.h5'
    print 'Reading ' + filename_in
    data = read_hdf(filename_in)
    data.slice(x3=127)
    data.shape
    #plotme(data)
    #show()
    filename_out = '/Users/mzoufras/Desktop/noprepulse/x2x1/x2x1-'+strnum+'.h5'
    #filename_out = '/Users/mzoufras/Desktop/PHA2/x2x1/x2x1-'+strnum+'.h5'
    print 'Writing ' + filename_out
    write_hdf(data,filename_out)
