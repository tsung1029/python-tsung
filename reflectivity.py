from h5_utilities import *
import matplotlib.pyplot as plt
import sys
import getopt
import glob
import numpy as np


def print_help():
    print 'reflectivity.py [options] <InputFile>'
    print 'options:'
    print '  --s1in: (can be a 1D array of 2n elements) Poynting flux coming in '
    print '  --interval: (1D array of 2n elements) interval of [on, off, ...], in the time unit of simulations'
    print '  --norm: (float) Poynting flux used in normalizing reflectivity'
    print '  --start: (int or array of int) over which indices to get averaged Poynting flux, default is [3,5]'
    print '  --smooth: (int) time steps to smooth, default is 3'


def moving_average(a, n=3):
    ret = np.cumsum(a)
    ret[n:] = ret[n:] - ret[:-n]
    return ret / n

argc = len(sys.argv)
try:
    opts, args = getopt.gnu_getopt(sys.argv[1:], "h", ['s1in=', 'norm=', 'interval=', 'start='])
except getopt.GetoptError:
    print_help()
    sys.exit(2)

start = range(3, 5)
smooth = 3
for opt, arg in opts:
    if opt in '--s1in':
        arg = arg.replace("(", "").replace(")", "").replace("[", "").replace("]", "")
        try:  # is an array like
            s1_base = np.asarray(eval(arg), dtype=float)
        except:  # is the file name of the s1 profile, same dimension as the data
            arg = arg.replace("'", "").replace('"', "")
            s1_base = read_hdf(str(arg)).data
    elif opt in '--interval':
        if opt.lower() != 'auto':
            arg = arg.replace("(", "").replace(")", "").replace("[", "").replace("]", "")
            # arg = '[' + arg + ']'
            interval = np.asarray(eval(arg), dtype=float)
        else:  # try to detect on/off automatically
            raise Exception('Not implemented!')
    elif opt in '--norm':
        s1_norm = eval(arg)
    elif opt in '--start':
        arg = arg.replace("(", "").replace(")", "").replace("[", "").replace("]", "")
        # arg = '[' + arg + ']'
        start = np.asarray(eval(arg), dtype=float)
    elif opt in '--smooth':
        smooth = eval(arg)
    else:
        print_help()
        sys.exit(2)

h5_data = read_hdf(args[0])
stlen = len(start)
try:
    s1_base
except NameError:
    try:
        s1_base = np.array([s1_norm])
    except NameError:
        s1_base = np.array([np.mean(h5_data.data[start[0]:start[stlen-1], :])])
try:
    interval
except NameError:
    interval = np.array(h5_data.axes[1].axis_max)  # default to a large number
period = np.sum(interval)
sec = np.cumsum(interval)
szp = np.size(interval)
if szp != np.size(s1_base):
    if np.size(s1_base) == np.size(h5_data.data):
        fac = np.sum(h5_data.data[2])/np.sum(s1_base[2])
        s1_base *= fac
        plt.plot(np.mean(h5_data.data, axis=1))
        plt.plot(np.mean(s1_base, axis=1))
        plt.show()
        s1_norm = np.mean(s1_base)
        h5_data.data -= s1_base
        s1_base = [0.0]
    else:
        raise Exception('Input error: dimension not match')
try:
    s1_norm
except NameError:
    s1_norm = np.sum(s1_base * interval) / sec[szp-1]
total_time = h5_data.shape[0]
h5_output = hdf_data()
# h5_output.data = np.zeros(total_time)
tn = range(total_time)
dt = (h5_data.axes[1].axis_max - h5_data.axes[1].axis_min) / h5_data.shape[0]
h5_output.data = np.zeros(h5_data.shape[0])
for ti in tn:
    time = (ti * dt) % sec[szp-1]
    for si in range(szp):
        if time < sec[si]:
            break
    h5_output.data[ti] = np.mean((s1_base[si] - h5_data.data[ti, :]) / s1_norm)
    # h5_output.data[ti] = np.mean(h5_data.data[ti, :])
    # h5_output.data[ti] = s1_base[si]
h5_output.data[0:start[0]] = 0.0
h5_output.shape = [total_time]
h5_output.axes = [h5_data.axes[1]]
h5_output.axes[0].axis_numberpoints = total_time
h5_output.axes[0].increment = dt

h5_output.data_attributes['LONG_NAME'] = ['Reflectivity']
h5_output.data_attributes['UNITS'] = ['Norm.']
# # try to remove artificial jumps in reflectivity due to averge process when obtaining poynting flux
for ti in range(start[0], total_time-1):
    if np.abs((2*h5_output.data[ti]-(h5_output.data[ti-1]+h5_output.data[ti+1]))/(h5_output.data[ti+1]-h5_output.data[ti-1])) > 10:
        h5_output.data[ti] = (h5_output.data[ti-1]+h5_output.data[ti+1]) * 0.5
ti = total_time-1
if (ti > 2) and np.abs((h5_output.data[ti]-h5_output.data[ti-1])/(h5_output.data[ti-1]-h5_output.data[ti-2])) > 3:
    h5_output.data[ti] = h5_output.data[ti-1]

h5_output.data = moving_average(h5_output.data, n=smooth)
plotme(h5_output)
pt = 0.8
print 'last', (1-pt)*100, '% time average =', np.mean(h5_output.data[int(pt*total_time):total_time-1])
write_hdf(h5_output, os.path.splitext(args[0])[0]+'-refl'+'.h5', dataset_name='S1', write_data=True)
h5_output.data = np.cumsum(h5_output.data) / np.arange(1, total_time+1)
print 'average =', h5_output.data[total_time-1]
write_hdf(h5_output, os.path.splitext(args[0])[0]+'-refl-avg'+'.h5', dataset_name='S1', write_data=True)
# plt.imshow(data)
plt.show()
