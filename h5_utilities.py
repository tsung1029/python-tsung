# Version 2:  September 2019  --> migrating to the PyVisOS library
#

import matplotlib.pyplot as plt
from matplotlib import cm, colors
import numpy as np
import h5py
import copy
# from pylab import *
import os



def fourier_in_space(data_bundle):
    # here we fourier analyze the data in space
    k_data=np.fft.fft(data_bundle.data,axis=1)
    k_data_2=np.fft.fft(k_data,axis=0)
    data_bundle.data=np.log(np.abs(k_data_2)+0.00000000001)

    dt=data_bundle.axes[0].max/(data_bundle.shape[0]-1)
    dx=data_bundle.axes[1].max/data_bundle.shape[1]
    data_bundle.axes[0].max=2.0*3.1415926/dt
    data_bundle.axes[1].max=2.0*3.1415926/dx

    return data_bundle


def plotme(hdfdata, data=None):
    data_to_use = hdfdata.data
    if data is not None:
        data_to_use = data

    if len(data_to_use.shape) == 1:
        plt.figure()
        plot_object = plt.plot(hdfdata.axes[0].get_axis_points(), data_to_use)
        plt.xlabel("%s \n %s" % (hdfdata.axes[0].attributes['LONG_NAME'][0],
                                 math_string(hdfdata.axes[0].attributes['UNITS'])[0]))
        plt.ylabel("%s \n %s" % (hdfdata.data_attributes['LONG_NAME'][0],
                                 math_string(hdfdata.data_attributes['UNITS'][0])))
        return plot_object

    if len(data_to_use.shape) == 2:
        extent_stuff = [hdfdata.axes[0].axis_min, hdfdata.axes[0].axis_max,
                        hdfdata.axes[1].axis_min, hdfdata.axes[1].axis_max]
        plot_object = plt.imshow(data_to_use, origin='lower', extent=extent_stuff, aspect='auto', cmap='Rainbow')
        cb = plt.colorbar(plot_object)
        cb.set_label("%s \n %s" % (hdfdata.data_attributes['LONG_NAME'], hdfdata.data_attributes['UNITS']))
        plt.xlabel("%s \n %s" % (hdfdata.axes[0].attributes['LONG_NAME'][0],
                                 math_string(hdfdata.axes[0].attributes['UNITS'])[0]))
        plt.ylabel("%s \n %s" % (hdfdata.axes[0].attributes['LONG_NAME'][0],
                                 math_string(hdfdata.axes[0].attributes['UNITS'][0])))
        # ylabel("%s \n %s"% ( hdfdata.data_attributes['LONG_NAME'], hdfdata.data_attributes['UNITS']) )


def math_string(input_str):
    try:
        input_str = input_str[0]
    except:
        pass
        input_str = str(input_str)
    return input_str


class hdf_data:
    def __init__(self):
        self.filename = None
        self.axes = []
        self.data = None
        self.data_attributes = {}
        self.run_attributes = {}
        self.shape = None

    def clone(self):
        out = copy.deepcopy(self)
        return out

    """
    def register_slice(self, data, list_indices_removed):
        temp_list = []
        for axis_index in range(0, len(self.axes)):
            if axis_index not in list_indices_removed:
                temp_list.append(item)
    """

    # full_expression.append(x_slice)
    def slice(self, x3=None, x2=None, x1=None, copy=False):
        slice_index_array = []
        # target_obj = self

        temp_axes = list(self.axes)
        for axis in temp_axes:
            # print "HI!!!!"
            selection = slice(None, None, None)
            # print "Gonna slice axis %d" % axis.axis_number
            if axis.axis_number == 1:
                selection = self.__slice_dim(x3, self.data, 3)
            if axis.axis_number == 2:
                selection = self.__slice_dim(x2, self.data, 2)
            if axis.axis_number == 3:
                selection = self.__slice_dim(x1, self.data, 1)

            slice_index_array.append(selection)

        # print slice_index_array
        new_data = self.data[slice_index_array]
        #
        #
        self.data = None
        self.data = new_data
        self.shape = new_data.shape

    def __slice_dim(self, indices, data, axis_direction):

        if indices is None:
            return slice(None, None, None)

        # a slicer tha tis just an interger
        #     means take a slice using that value.
        #    so remove the coresponding axis.
        if isinstance(indices, int):
            if (len(self.axes) > 1):
                self.__remove_axis(axis_direction)
            return indices

        array_slice = None
        # we accept lots of different specifications for the indecies.. let's sort it out.
        if (indices is not None):
            # now see if a python list was passed in.
            # if so, make it into a slicer
            try:
                size = len(indices)
                if size == 0:
                    array_slice = slice(None, None, None)
                elif size == 1:
                    array_slice = slice(indices[0], None, None)
                elif size == 2:
                    array_slice = slice(indices[0], indices[1], None)
                elif size == 3:
                    array_slice = slice(indices[0], indices[1], indices[2])
            except:
                # Well, what ever is there, just try to use it like a slicer
                #        if it works, then go with it.
                array_slice = indices

            try:
                # These next lines just test that we have a 'slicer'
                #        (or functional equivilent). If these fileds are not
                #        there, then an exception will be thrown.
                # temp = array_slice.start
                # temp = array_slice.stop
                # temp = array_slice.end

                # ------START LOGIC---------------
                # by this time, we have processed all different input types
                # into a standard form: a slicer object named 'indices'

                # a slicer with start=None and end=None means take the whole axis as-is.
                if array_slice.start is None and array_slice.stop is None:
                    return array_slice

                axis = self.get_axis(axis_direction)

                new_start_index = 0
                new_stop_index = len(data)
                if array_slice.start is not None:
                    new_start_index = array_slice.start
                if array_slice.stop is not None:
                    new_stop_index = array_slice.stop

                # now update the axis's data to reflect the new slice start/stop.
                # update the Python side.
                axis.axis_min = (axis.increment * new_start_index) + axis.axis_min
                axis.axis_max = (axis.increment * new_stop_index) + axis.axis_min
                self.axis_numberpoints = new_stop_index - new_stop_index

                # update to dictionary 'attributues' will happen in the
                # 'write_hdf' function.,.. just to keep all HDF action localised to
                # isolated routines.
                return array_slice

                # TODO: handle the Elipsis object..
                # also can use test[ix_(a_dim,b_dim,c_dim)] but not quite flexable enough?
            except:
                # broken slicer..
                raise Exception("Invalid indices for array slice")

        return slice(None, None, None)

    # return None if axis is not present.
    def get_axis(self, axis_index):
        for (i, axis) in enumerate(self.axes):
            if axis.axis_number == axis_index:
                return axis
        return None

    def __remove_axis(self, axis_index):
        for (i, axis) in enumerate(self.axes):
            if axis.axis_number == axis_index:
                del self.axes[i]
        pass

    def remove_axis(self, axis_index):
        self.__remove_axis(axis_index)

    def __axis_exists(self, axis_index):
        for (i, axis) in enumerate(self.axes):
            if axis.axis_number == axis_index:
                return True
        return False


class data_basic_axis:
    def __init__(self, axis_number, axis_min, axis_max, axis_numberpoints):
        self.axis_number = axis_number
        self.axis_min = axis_min
        self.axis_max = axis_max
        self.axis_numberpoints = axis_numberpoints
        self.increment = (self.axis_max - self.axis_min) / self.axis_numberpoints
        self.attributes = {}

    def clone(self):
        out = copy.deepcopy(self)
        return out

    def get_axis_points(self):
        return np.arange(self.axis_min, self.axis_max, self.increment)

    def trim(self, n_pts_start, n_pts_end):
        self.axis_numberpoints -= (n_pts_start + n_pts_end)
        if self.axis_numberpoints < 0:
            print('ERROR: empty axis!')
        self.axis_min += n_pts_start * self.increment
        self.axis_max -= n_pts_end * self.increment



# ----------------------------------------------------------------------------------------------------------
#                    Setup maps etc...
# ----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------
idl_13_r = [0, 4, 9, 13, 18, 22, 27, 31, 36, 40, 45, 50, 54, 58, 61, 64, 68, 69, 72, 74, 77, 79, 80, 82, 83, 85, 84, 86,
            87, 88, 86, 87, 87, 87, 85, 84, 84, 84, 83, 79, 78, 77, 76, 71, 70, 68, 66, 60, 58, 55, 53, 46, 43, 40,
            36, 33, 25, 21, 16, 12, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 8, 12, 21, 25, 29,
            33, 42, 46, 51, 55, 63, 67, 72, 76, 80, 89, 93, 97, 101, 110, 114, 119, 123, 131, 135, 140, 144, 153, 157,
            161,
            165, 169, 178, 182, 187, 191, 199, 203, 208, 212, 221, 225, 229, 233, 242, 246, 250, 255, 255, 255, 255,
            255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
            255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
            255, 255, 255, 255]
idl_13_g = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 8, 16, 21, 25, 29, 38, 42, 46, 51, 55, 63,
            67, 72, 76, 84, 89, 93, 97, 106, 110, 114,
            119, 127, 131, 135, 140, 144, 152, 157, 161, 165, 174, 178, 182, 187, 195, 199, 203, 208, 216, 220, 225,
            229, 233, 242, 246, 250, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
            255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
            255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
            255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
            255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 250, 242,
            238, 233, 229, 221, 216, 212, 208, 199, 195, 191, 187, 178, 174, 170, 165, 161, 153, 148, 144, 140, 131,
            127, 123, 119, 110, 106, 102, 97, 89, 85, 80, 76, 72, 63, 59, 55, 51, 42, 38, 34, 29, 21,
            17, 12, 8, 0]
idl_13_b = [0, 3, 7, 10, 14, 19, 23, 28, 32, 38, 43, 48, 53, 59, 63, 68, 72, 77, 81, 86, 91, 95, 100, 104, 109, 113,
            118, 122, 127, 132, 136, 141, 145, 150, 154, 159, 163, 168, 173, 177, 182, 186,
            191, 195, 200, 204, 209, 214, 218, 223, 227, 232, 236, 241, 245, 250, 255, 255, 255, 255, 255, 255, 255,
            255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
            255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
            255, 255, 255, 255, 255, 255, 246, 242, 238, 233, 225, 220, 216, 212, 203, 199, 195, 191, 187, 178, 174,
            170, 165, 157, 152, 148, 144, 135, 131, 127, 123, 114, 110, 106, 102, 97, 89, 84, 80, 76, 67, 63, 59, 55,
            46, 42, 38, 34, 25, 21, 16, 12, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0]



def init_colormap():
    rgb = []
    for i in range(0, len(idl_13_r)):
        rgb.append((float(idl_13_r[i]) / 255.0, float(idl_13_g[i]) / 255.0, float(idl_13_b[i]) / 255.0))
    color_map = colors.ListedColormap(rgb, name='Rainbow')
    color_map.set_under(rgb[0])
    color_map.set_over(rgb[-1])
    cm.register_cmap(name='Rainbow', cmap=color_map)


init_colormap()


"""
        
    A Minimum is assumed about the HDF5 structure. What is assumed is:
        0.     There is an attribute in the root of the hdf5 named 'NAME'
                (you would access it with: h5py_file['NAME']). This contains the
                array entry of the that holds the main data.. so say h5py_file['NAME']) = 'p1x1'. 
                Then the main data array will be under the name 'p1x1'. 
                You could get with:    h5py_file['p1x1'].value
        
        1.        In the HDF, the is a group (a group is just a folder in MAC osx) named 'AXIS'.
                Inside that folder sits the info about each axis in the HDF dataset. For 1D data,
                there is 1 entry: AXIS/AXIS1. If you get the array data here 
                using:        h5py_file['AXIS/AXIS1'].value 
                It will be a 1D array with only 2 values; these are the min/max of the axis when plotting.
                Each axis also had some attributes. The 4 usual ones are TYPE,UNITS,NAME,LONG_NAME (type
                is linear or log etc).. Others are string that can be used for axis labels. In 2D there are
                two entries under AXIS folder and 3 for 3D (h5py_file['AXIS/AXIS1'], h5py_file['AXIS/AXIS2'],
                h5py_file['AXIS/AXIS3'])
        2. We need 1 more 'axis' (no matter in 1d, 2d, or 3d). It is this access that will decribe the actual
                quantity being measued (E.g. the veritcal axis in most 1-D plots... the height or color of a region in
                a 2D etc.. This indo is in attributes attched to the main data array we found in Step 0.
                It has 2 attributes: UNITS and LONG NAME for labeling the data when plotted.
        3. The root has lots of attributes. It has stuff about time:
                    ITER = iteration data is showing. DT=step size used, 
                    TIME=simualtion time data is showing, TIME_UNITS=string.. good label to put on plots.
                    "MOVE C" = [x, y, z] where 0 for a componet means that component isn'i interested 
                                    and a 1 means the compent wants to/or have done this ("MOVE C" tells us moving
                                    window directions. PERODIC is same idea as "MOVE C".. just showing perodic directions.
                    "XMIN" = [x_min, y_min, z_min] and "XMAX" = [x_max, y_max, z_max].. 
                    always 3 compents.
        4. ALL DATA/ATTRIBUTES ARE COPIED OUT OF HDF INTO MEMORY. Then HDF file is closed. There are other ways.. and for
            very large files that wont fit in memory, other options will have to be used that allow 'streaming' of the 
            input files.
                

"""
