import matplotlib
matplotlib.use('Agg')
import multiprocessing, h5py, sys, glob, os, shutil, subprocess
import numpy as np

from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as ticker
from h5_utilities import *
from analysis import analysis
import re
from str2keywords import str2keywords
# global parameters


section_start, section_end, = '{', '}'
ignore_flag, empty, eq_flag, tokenize_flag = '!', '', '=', ','
general_flag, subplot_flag = 'simulation', 'data'
cpu_count = multiprocessing.cpu_count()
# match commas not inside single or double quotes
option_pattern = re.compile(''',(?=(?:(?:[^"']*"[^"']*")|(?:[^'"]*'[^'"]*'))*[^"']*$)''')


def main():
    args = sys.argv
    print 'num cores: ' + str(cpu_count)
    file = open(args[1], 'r')
    input_file = file.read()
    file.close()
    # read input deck general parameters
    plots = Plot(input_file)
    dirs = plots.general_dict['save_dir'][0]
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    # else:
    # 	shutil.rmtree(dirs)
    # 	os.makedirs(dirs)
    dpi = plots.general_dict['dpi'][0]
    fig_size = plots.general_dict['fig_size']
    x, y = dpi * fig_size[0], dpi * fig_size[1]
    if (x * y > 4000 * 2000):
        x, y = x / 2, y / 2
    plots.parallel_visualize()

    if (dirs[len(dirs) - 1] != '/'):
        dirs = dirs + '/'
    stdout = subprocess.check_output(['ffmpeg', '-encoders', '-v', 'quiet'])
    for encoder in ['libx264', 'mpeg4', 'mpeg']:
        if encoder in stdout:
            break
    else:
        return  # unsupported encoder
    subprocess.call(
        ["ffmpeg", "-framerate", "10", "-pattern_type", "glob", "-i", dirs + '*.png', '-c:v', encoder, '-vf',
         'scale=' + str(x) + ':' + str(y) + ' ,format=yuv420p', '-y', dirs + 'movie.mp4'])


def get_bounds(self, file_name, num, file_num):
    file_name = file_name + str(str(1000000 + num)[1:]) + '.h5'
    if (os.path.isfile(file_name)): 
        try:
            file = h5py.File(file_name, 'r')
            data = self.get_data(file, file_num)
        except:
            data = read_hdf(file_name).data
        if ('operation' in self.general_dict.keys()):
            data = analysis(data, self.general_dict.get('operation'))
        minimum, maximum = np.min(data), np.max(data)
        file.close()
        del data
    else:
        minimum,maximum = 0.0, 0.0
    return minimum, maximum


def fmt(x, pos):
    a, b = '{:.2e}'.format(x).split('e')
    b = int(b)
    a = float(a)
    if (a == 0):
        return '0'
    if (a < 0):
        x = '-'
    else:
        x = ''

    return x + '$\mathregular{' + '10^{{{}}}'.format(b) + '}$'


def visualize(plot, indices):
    subplots = plot.subplots
    title = ''
    for num in xrange(len(subplots)):
        height, width = plot.general_dict['fig_size']
        fig = plt.figure(1, figsize=(height, width))
        out_title = subplots[num].graph(fig, indices, num + 1)
        title = max([title, out_title], key=len)
    plt.suptitle(title, fontsize=plot.general_dict['fontsize'][0] * 1.2)
    fol = plot.general_dict['save_dir'][0]
    if (fol[len(fol) - 1] != '/'):
        fol = fol + '/'
    plt.savefig(fol + str(indices + 1000000)[1:], dpi=plot.general_dict['dpi'][0], bbox_inches='tight')
    plt.close()


class Plot:
    def __init__(self, text):
        # if you want extra parameters at start modify self.types
        self.types = {'subplots': int, 'nstart': int, 'nend': int, 'ndump': int, \
                      'dpi': int, 'fig_size': float, 'fig_layout': int, 'fontsize': int, 'save_dir': str,
                      'sim_dir': str}
        # size in inches, dpi for image quality, configuration for plot layouts


        self.flag = general_flag
        self.general_keys = self.types.keys()
        self.general_dict = {}
        self.subplots = []
        self.read_general_parameters(text, 0)
        for num in xrange(self.general_dict['subplots'][0]):
            self.subplots.append(Subplot(text, num, self.general_dict))

    def read_general_parameters(self, text, ind):
        # string = text.lower()
        string = self.find_section(text, ind, self.flag)
        self.read_lines(string)

    def find_section(self, text, ind, keyword):
        lines = text.splitlines()
        ind_pas = 0
        start = 0
        end = 0
        kwlen = len(keyword)
        for line in lines:
            if (ignore_flag in line):
                curr_line = line[:line.find(ignore_flag)].lower()
            else:
                curr_line = line.lower().lstrip()
            if (keyword == curr_line[0:kwlen]):
                if (ind_pas == (ind + 1)):
                    break
                else:
                    ind_pas += 1
            if (ind_pas <= ind):
                start += len(line) + 1
            end += len(line) + 1
        sub_text = text[start:end]
        sub_lines = sub_text.splitlines()
        return sub_text[sub_text.find(section_start):sub_text.rfind(section_end) + 1]

    def read_lines(self, string):
        lines = string.splitlines()
        for line in lines:
            if (ignore_flag in line):
                line = line[:line.find(ignore_flag)]
            for key in self.general_keys:
                if (key in line.lower()):
                    self.general_dict[key] = self.tokenize_line(line, self.types[key])

    def tokenize_line(self, str, cast_type):
        start = str.find(eq_flag) + 1
        # str = str[start:].split(tokenize_flag)
        str = option_pattern.split(str[start:])
        out = []
        for s in str:
            s = s.strip()
            if (s != empty):
                ele = s.strip("'").strip("\"")
                if (s == 'None' or s == 'none'):
                    out.append('None')
                else:
                    out.append(cast_type(ele))
        return out

    def parallel_visualize(self):
        nstart, ndump, nend = self.general_dict['nstart'], self.general_dict['ndump'], self.general_dict['nend']
        total_num = (np.array(nend) - np.array(nstart)) / np.array(ndump)
        Parallel(n_jobs=cpu_count)(delayed(visualize)(self, nn) for nn in xrange(np.min(total_num) + 1))


class Subplot(Plot):
    def __init__(self, text, num, general_params):
        # if you want extra parameters in subplots -- modify self.types
        self.params = general_params
        self.types = {'folders': str, 'title': str, 'log_threshold': float, \
                      'plot_type': str, 'maximum': float, 'minimum': float, \
                      'colormap': str, 'legend': str, 'markers': str, \
                      'x1_zoom': float, 'x2_zoom': float, 'x3_zoom': float, 'norm': str, 'side': str, 'bounds': str, \
                      'use_dir': str, 'linewidth': float, 'operation': str2keywords}
        self.left = 0
        self.right = 0
        self.general_dict = {}
        self.raw_edges = {}
        self.file_names = []
        self.general_keys = self.types.keys()
        self.flag = subplot_flag
        self.read_general_parameters(text, num)
        self.get_file_names()
        self.count_sides()
        self.set_limits()
        print self.general_dict
        print self.raw_edges

    def count_sides(self):
        if ('side' in self.general_dict.keys()):
            for j in self.general_dict['side']:
                if (j == 'left'):
                    self.left += 1
                else:
                    self.right += 1
        else:
            self.general_dict['side'] = ['left'] * len(self.general_dict['folders'])
            self.left = len(self.general_dict['folders'])
            self.right = 0

    def get_file_names(self):
        folders = self.general_dict['folders']
        for index in xrange(len(folders)):
            folder = folders[index]
            if ('use_dir' not in self.general_dict.keys() or (
                (index < len(self.general_dict['use_dir'])) and self.general_dict['use_dir'][index] == 'True')):
                if ('sim_dir' in self.params.keys()):
                    if (index < len(self.params['sim_dir'])):
                        sim_dir = self.params['sim_dir'][index]
                    else:
                        if (folder[0:2] == 'MS' or folder[1:3] == 'MS'):
                            sim_dir = self.params['sim_dir'][len(self.params['sim_dir']) - 1]
                        else:
                            sim_dir = ''
                    if (len(sim_dir) > 0 and sim_dir[len(sim_dir) - 1] != '/'):
                        folder = sim_dir + '/' + folder
                    else:
                        folder = sim_dir + folder
            print folder
            if (folder[len(folder) - 1] == '/'):
                new2 = glob.iglob(folder + '*.h5')
            else:
                new2 = glob.iglob(folder + '/*.h5')
            first = next(new2)
            first = first[:len(first) - 9]  # removes 000000.h5
            self.file_names.append(first)

    def get_nfac(self, index):
        if (index < len(self.params['nstart'])):
            return self.params['nstart'][index], self.params['ndump'][index], self.params['nend'][index]
        else:
            last_ind = len(self.params['nstart']) - 1
            return self.params['nstart'][last_ind], self.params['ndump'][last_ind], self.params['nend'][last_ind]

    def set_limits(self):
        folders = self.general_dict['folders']
        subplot_keys = self.general_dict.keys()
        minimum, maximum = None, None
        min_max_pairs = []

        for index in xrange(len(folders)):
            nstart, ndump, nend = self.get_nfac(index)

            out = Parallel(n_jobs=cpu_count)(
                delayed(get_bounds)(self, self.file_names[index], nn, index) for nn in xrange(nstart, nend + 1, ndump))
            print out
            for mn, mx in out:
                if (maximum == None):
                    maximum = mx
                else:
                    if (mx > maximum):
                        maximum = mx
                if (minimum == None):
                    minimum = mn
                else:
                    if (mn < minimum):
                        minimum = mn
            min_max_pairs.append((minimum, maximum))
            maximum, minimum = None, None
        mins, maxs = [np.inf, np.inf], [-np.inf, -np.inf]
        for file_num in xrange(len(folders)):
            mn, mx = min_max_pairs[file_num]
            if (self.general_dict['side'][file_num] == 'left'):
                mins[0] = min(mins[0], mn)
                maxs[0] = max(maxs[0], mx)
            else:
                mins[1] = min(mins[1], mn)
                maxs[1] = max(maxs[1], mx)
        if ('maximum' in self.general_dict.keys()):
            for ind in xrange(len(self.general_dict['maximum'])):
                if (self.general_dict['maximum'] != 'None'):
                    maxs[ind] = self.general_dict['maximum'][ind]
        if ('minimum' in self.general_dict.keys()):
            for ind in xrange(len(self.general_dict['minimum'])):
                if (self.general_dict['minimum'] != 'None'):
                    mins[ind] = self.general_dict['minimum'][ind]

        self.general_dict['minimum'] = mins
        self.general_dict['maximum'] = maxs
        print mins, maxs

    def get_min_max(self, file_num):
        if (self.general_dict['side'][file_num] == 'left'):
            return self.general_dict['maximum'][0], self.general_dict['minimum'][0]
        else:
            return self.general_dict['maximum'][1], self.general_dict['minimum'][1]

    def graph(self, figure, n_ind, subplot_num):
        fig = figure
        rows, columns = self.params['fig_layout']
        ax_l = plt.subplot(rows, columns, subplot_num)
        ax = ax_l
        time = r'$ t = '

        plot_prev = None
        len_file_names = 0
        for j in xrange(2):
            if (j == 0):
                key = 'left'
            else:
                key = 'right'
                if (self.right == 0):
                    break
                else:
                    ax = ax_l.twinx()
            for file_num in xrange(len(self.file_names)):
                if (self.general_dict['side'][file_num] == key):
                    nstart, ndump, nend = self.get_nfac(file_num)
                    nn = ndump * n_ind + nstart
                    file_name = self.file_names[file_num] + str(int(1000000 + nn))[1:] + '.h5'
		    if (os.path.isfile(file_name)):
                        file = h5py.File(self.file_names[file_num] + str(int(1000000 + nn))[1:] + '.h5', 'r')
                        plot_type = self.get_indices(file_num)[0]
    
                        if (plot_type == 'slice'):
                            self.plot_grid(file, file_num, ax, fig)
                        elif (plot_type == 'raw'):
                            self.plot_raw(file, file_num, ax, fig)
                        elif (plot_type == 'lineout'):
                            self.plot_lineout(file, file_num, ax, fig)

                        plot_prev = plot_type
                        time = time + str(file.attrs['TIME'][0])
                        if (file_num < len(self.file_names) - 1):
                            time = time + ','
                        file.close()
            self.set_legend(key, ax)

        time = time + '$'
        return time

    def set_legend(self, key, ax):
        select = {'left': self.left, 'right': self.right}
        num_on_axis = select[key]
        if (num_on_axis > 1):
            if (key == 'right'):
                ax.legend(loc=1)
            else:
                ax.legend(loc=2)

    def add_colorbar(self, imAx, label, ticks, ax, fig):
        plt.minorticks_on()
        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes("right", size="2%", pad=0.05)
        # plt.colorbar(imAx,cax=cax,format='%.1e')
        # ax.minorticks_on()
        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes("right", size="2%", pad=0.05)
        if (ticks == None):
            cb = fig.colorbar(imAx, pad=0.075)
            cb.set_label(label, fontsize=self.fontsize())
            cb.formatter.set_powerlimits((0, 0))
            cb.update_ticks()
        else:
            cb = fig.colorbar(imAx, pad=0.075, ticks=ticks, format=ticker.FuncFormatter(fmt))
            cb.set_label(label, fontsize=self.fontsize())



        # cb.ticklabel_format(style='sci', scilimits=(0,0))

    def mod_tickers(self, minimum, maximum, threshold):
        out = []
        epsilon = 1e-100
        mu = int(np.log10(np.abs(maximum + epsilon))) + 1
        mx_sign, mn_sign = 1, -1
        if (maximum < 0):
            mx_sign = mx_sign * -1
        if (minimum > 0):
            mn_sign = mn_sign * -1
        ml = int(np.log10(threshold)) - 1
        ll = int(np.log10(np.abs(minimum + epsilon))) + 1
        dx = -1
        if (ll - ml > 4):
            dx = -2
        for j in xrange(ll, ml, dx):
            out.append(mn_sign * 10 ** (j))
        out.append(0)
        dx = 1
        if (mu - ml > 4):
            dx = 2
        for j in xrange(ml + 1, mu + 1, dx):
            out.append(mx_sign * 10 ** (j))
        return out

    def get_indices(self, file_num):
        ctr = 0
        index = 0
        index_start = 0
        plot_types = self.general_dict['plot_type']
        while (True):
            if (index == len(plot_types)):
                break
            if (plot_types[index].lower() in ['slice', 'lineout', 'raw']):
                if (ctr == (file_num + 1)):
                    break
                ctr += 1
                index_start = index
            index += 1
        return self.general_dict['plot_type'][index_start:index]

    def get_data(self, file, file_num):

        indices = self.get_indices(file_num)
        if (len(indices) > 1):
            selectors = indices[1:]
        else:
            selectors = None
        plot_type = indices[0]
        axis_labels = []
        if ('axes' not in self.general_dict.keys()):
            self.general_dict['axes'] = []
        if (plot_type == 'slice' or plot_type == 'lineout'):
            axes = file['AXIS']
            for j in axes.keys():
                axis_labels.append(axes[j].attrs['NAME'][0])

            if (selectors == None):
                self.general_dict['axes'].extend(axis_labels)
                data = (file[file.attrs['NAME'][0]][:])
                return data

        if (plot_type == 'slice'):
            axis_labels.remove(selectors[0])
            self.general_dict['axes'].extend(axis_labels)
            data = (file[file.attrs['NAME'][0]][:])
            if (selectors[0] == 'x1'):
                return data[:, :, int(selectors[1])]
            elif (selectors[0] == 'x2'):
                return data[:, int(selectors[1]), :]
            else:
                return data[int(selectors[1]), :, :]

        if (plot_type == 'lineout'):
            self.general_dict['axes'].append(selectors[0])
            data = (file[file.attrs['NAME'][0]][:])
            if (len(selectors) == 3):
                x2_ind, x3_ind = int(selectors[1]), int(selectors[2])
                if (selectors[0] == 'x1'):
                    return data[x3_ind, x2_ind, :]
                elif (selectors[0] == 'x2'):
                    return data[x3_ind, :, x2_ind]
                else:
                    return data[:, x3_ind, x1_ind]
            else:
                x2_ind = int(selectors[1])
                if (selectors[0] == 'x1'):
                    return data[x2_ind, :]
                else:
                    return data[:, x2_ind]
        if (plot_type == 'raw'):
            bins = int(selectors[-1])
            if (len(self.general_dict['axes']) <= file_num):
                self.general_dict['axes'].extend(selectors[:-1])
            dim = len(selectors[:-1])
            if (len(file['q'].shape) == 0):
                print file['q'].shape
                if (dim == 2):
                    self.raw_edges[file_num] = [np.zeros(1), np.zeros(1)]
                    return np.zeros(1)
                else:
                    self.raw_edges[file_num] = [np.zeros(1)]
                    return np.zeros(1)

            q_weight = file['q'][:]
            nx = file.attrs['NX'][:]
            dx = (file.attrs['XMAX'][:] - file.attrs['XMIN'][:]) / (nx)
            if ('norm' in self.general_dict.keys() and file_num < len(self.general_dict['norm']) and
                        self.general_dict['norm'][file_num] == 'cylin'):
                norm = np.pi * 2 * np.prod(dx)
            else:
                norm = np.prod(dx)
            print np.sum(q_weight * norm), 'charge', self.file_names[file_num], norm
            print len(q_weight), 'length'
            if (dim == 2):
                if (selectors[0] == 'r'):
                    data1 = ((file['x2'][:]) ** 2 + (file['x3'][:]) ** 2) ** (0.5)
                else:
                    data1 = file[selectors[0]][:]
                if (selectors[1] == 'r'):
                    data2 = ((file['x2'][:]) ** 2 + (file['x3'][:]) ** 2) ** (0.5)
                else:
                    data2 = file[selectors[1]][:]
                bounds1 = self.get_bounds(selectors[0])
                bounds2 = self.get_bounds(selectors[1])
                if (bounds1 == None or bounds2 == None):
                    bounds = None
                else:
                    bounds = [bounds1, bounds2]
                hist, yedges, xedges = np.histogram2d(data1, data2, bins=bins, range=bounds,
                                                      weights=np.abs(q_weight * norm))
                xedges, yedges = (xedges[1:] + xedges[:-1]) / 2.0, (yedges[1:] + yedges[:-1]) / 2.0
                self.raw_edges[file_num] = [xedges, yedges]
            else:
                if (selectors[0] == 'r'):
                    data1 = ((file['x2'][:]) ** 2 + (file['x3'][:]) ** 2) ** (0.5)
                else:
                    data1 = file[selectors[0]]
                bounds = self.get_bounds(selectors[0])
                hist, bin_edges = np.histogram(data1, bins=bins, range=bounds, weights=np.abs(q_weight * norm))
                bin_edges = (bin_edges[1:] + bin_edges[:-1]) / 2.0
                self.raw_edges[file_num] = [bin_edges]
            return hist

    def append_legend(self, file_num):
        if ('legend' in self.general_dict.keys() and file_num < len(self.general_dict['legend'])):
            return r'${}$'.format(self.general_dict['legend'][file_num])
        else:
            return ''

    def get_bounds(self, label):
        if ('bounds' in self.general_dict.keys() and label in self.general_dict['bounds']):
            bounds = self.general_dict['bounds']
            index = bounds.index(label) + 1
            return map(float, bounds[index:(index + 2)])
        else:
            return None

    def plot_lineout(self, file, file_num, ax, fig):
        try:
            data = self.get_data(file, file_num)
        except:
            data = read_hdf(file.filename).data
        if ('operation' in self.general_dict.keys()):
            data = analysis(data, self.general_dict['operation'])
        axes = self.get_axes(file_num)
        xx = self.construct_axis(file, axes[0], file_num)
        maximum, minimum = self.get_min_max(file_num)
        indices = self.get_indices(file_num)
        selectors = indices[1:-1]
        if (indices[0].lower() == 'raw'):
            label = self.get_name(file, 'q') + self.append_legend(file_num)
        else:
            label = self.get_name(file) + self.append_legend(file_num)

        if (self.is_log_plot(file_num)):
            ax.plot(xx, data, self.get_marker(file_num), label=label, linewidth=self.get_linewidth())
            side = self.general_dict['side'][file_num]
            if (side == 'left'):
                ind = 0
            else:
                ind = 1
            threshold = self.general_dict['log_threshold'][ind]
            plt.yscale('symlog', linthreshy=threshold, vmin=minimum, vmax=maximum)
            ax.set_ylim([minimum, maximum])

        else:
            ax.plot(xx, data, self.get_marker(file_num), label=label, linewidth=self.get_linewidth())
            ax.set_ylim([minimum, maximum])
            ax.minorticks_on()
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

        self.set_labels(ax, file, axes, file_num)

    def get_linewidth(self):
        if ('linewidth' in self.general_dict.keys()):
            return self.general_dict['linewidth'][0]
        else:
            return 1

    def plot_raw(self, file, file_num, ax, fig):
        indices = self.get_indices(file_num)
        if (len(indices) > 1):
            selectors = indices[1:]
        else:
            selectors = None
        dim = len(selectors[:-1])
        if (len(self.get_data(file, file_num)) == 1):
            return
        if (dim == 2):
            self.plot_grid(file, file_num, ax, fig)
        else:
            self.plot_lineout(file, file_num, ax, fig)

    def plot_grid(self, file, file_num, ax, fig):
        try:
            data = self.get_data(file, file_num)
        except:
            data = read_hdf(file.filename).data
        if ('operation' in self.general_dict.keys()):
            data = analysis(data, self.general_dict['operation'])
        axes = self.get_axes(file_num)
        axis1 = self.construct_axis(file, axes[0], file_num)
        axis2 = self.construct_axis(file, axes[1], file_num)
        grid_bounds = [axis1[0], axis1[-1], axis2[0], axis2[-1]]

        maximum, minimum = self.get_min_max(file_num)

        if (self.is_log_plot(file_num)):
            if (maximum == 0):
                new_max = 0
            else:
                new_max = maximum / np.abs(maximum) * 10 ** (int(np.log10(np.abs(maximum))) + 1)
            if (minimum == 0):
                new_min = 0
            else:
                new_min = minimum / np.abs(minimum) * 10 ** (int(np.log10(np.abs(minimum))) + 1)

            threshold = self.general_dict['log_threshold'][file_num]
            imAx = ax.imshow(data, aspect='auto', origin='lower', \
                             interpolation='bilinear', vmin=new_min, vmax=new_max, \
                             norm=matplotlib.colors.SymLogNorm(threshold), extent=grid_bounds,
                             cmap=self.get_colormap(file_num))
        else:
            imAx = ax.imshow(data, aspect='auto', origin='lower', \
                             interpolation='bilinear', vmin=minimum, vmax=maximum, extent=grid_bounds,
                             cmap=self.get_colormap(file_num))

        if ('x1_zoom' in self.general_dict.keys()):
            plt.xlim(self.general_dict['x1_zoom'])
        if ('x2_zoom' in self.general_dict.keys()):
            plt.ylim(self.general_dict['x2_zoom'])
        indices = self.get_indices(file_num)
        selectors = indices[1:-1]
        if (indices[0].lower() == 'raw'):
            long_name = self.get_name(file, 'q') + self.append_legend(file_num) + r'$\/$' + self.get_units(file, 'q')
        else:
            long_name = self.get_name(file) + self.append_legend(file_num) + r'$\/$' + self.get_units(file)

        if (self.is_log_plot(file_num)):
            self.add_colorbar(imAx, long_name, self.mod_tickers(minimum, maximum, threshold), ax, fig)
        else:
            self.add_colorbar(imAx, long_name, None, ax, fig)

        self.set_labels(ax, file, axes, file_num)

    # plt.title(,fontsize = self.fontsize())


    def set_labels(self, ax, file, axes, file_num):
        select = {'left': self.left, 'right': self.right}
        num_on_axis = select[self.general_dict['side'][file_num]]
        plot_type = self.get_indices(file_num)[0]
        if (plot_type == 'lineout'):
            if (num_on_axis == 1):
                ax.set_xlabel(self.axis_label(file, axes[0]), fontsize=self.fontsize())
                ax.set_ylabel(self.get_long_name(file), fontsize=self.fontsize())
            else:
                ax.set_xlabel(self.axis_label(file, axes[0]), fontsize=self.fontsize())
                ax.set_ylabel(self.get_units(file), fontsize=self.fontsize())
        elif (plot_type == 'slice'):
            ax.set_xlabel(self.axis_label(file, axes[0]), fontsize=self.fontsize())
            ax.set_ylabel(self.axis_label(file, axes[1]), fontsize=self.fontsize())
        elif (plot_type == 'raw'):
            indices = self.get_indices(file_num)
            selectors = indices[1:-1]
            dim = len(selectors)
            if (dim == 2):
                ax.set_xlabel(self.axis_label(file, axes[0], selectors[1]), fontsize=self.fontsize())
                ax.set_ylabel(self.axis_label(file, axes[1], selectors[0]), fontsize=self.fontsize())
            else:
                if (num_on_axis == 1):
                    label = self.get_long_name(file, 'q')
                    ax.set_xlabel(self.axis_label(file, axes[0], selectors[0]), fontsize=self.fontsize())
                    ax.set_ylabel(label, fontsize=self.fontsize())
                else:
                    ax.set_xlabel(self.axis_label(file, axes[0], selectors[0]), fontsize=self.fontsize())
                    ax.set_ylabel(self.get_units(file, 'q'), fontsize=self.fontsize())

    def get_marker(self, file_num):
        if ('markers' in self.general_dict.keys() and file_num < len(self.general_dict['markers'])):
            return self.general_dict['markers'][file_num]
        else:
            return ''

    def get_units(self, file, keyword=None):
        ## assuming osiris notation
        if (keyword == None):
            try:
                data = file[file.attrs['NAME'][0]]
            except:
                return '[a.u.]'
        else:
            data = file[keyword]
        UNITS = data.attrs['UNITS'][0]
        return r'$[{}]$'.format(UNITS)

    def get_name(self, file, keyword=None):
        ## assuming osiris notation
        if (keyword == None):
            try:
                data = file[file.attrs['NAME'][0]]
            except:
                return r'${}$'.format(file.attrs['NAME'][0])
        else:
            data = file[keyword]
        NAME = data.attrs['LONG_NAME'][0]
        return r'${}$'.format(NAME)

    def get_long_name(self, file, keyword=None):
        ## assuming osiris notation
        if (keyword == None):
            data = file[file.attrs['NAME'][0]]
        else:
            data = file[keyword]
        UNITS = data.attrs['UNITS'][0]
        NAME = data.attrs['LONG_NAME'][0]
        return r'${}\/[{}]$'.format(NAME, UNITS)

    def get_axes(self, file_num):
        count, nn = 0, 0
        for num in xrange(file_num):
            indices = self.get_indices(num)
            typex = indices[0]
            if (typex == 'slice'):
                count += 2
            elif (typex == 'lineout'):
                count += 1
            if (typex == 'raw'):
                count += len(indices[1:]) - 1

        if (self.get_indices(file_num)[0] == 'slice'):
            nn = 2
        elif (self.get_indices(file_num)[0] == 'lineout'):
            nn = 1
        elif (self.get_indices(file_num)[0] == 'raw'):
            nn = len(self.get_indices(file_num)[1:]) - 1
        print self.general_dict['axes']
        print count, count + nn
        return self.general_dict['axes'][count:(count + nn)]

    def fontsize(self):
        if ('fontsize' in self.params.keys()):
            return self.params['fontsize'][0]
        return 16

    def get_colormap(self, file_num):
        if ('colormap' in self.general_dict.keys()):
            return self.general_dict['colormap'][file_num]
        else:
            return None

    def is_log_plot(self, file_num):
        side = self.general_dict['side'][file_num]
        if (side == 'left'):
            ind = 0
        else:
            ind = 1
        return 'log_threshold' in self.general_dict.keys() and type(self.general_dict['log_threshold'][ind]) is float

    def construct_axis(self, file, label, file_num):
        ## assuming osiris notation
        indices = self.get_indices(file_num)
        selectors = indices[1:]
        if (indices[0] == 'raw'):
            print file_num, label, selectors[0]
            if (label == selectors[0]):
                return self.raw_edges[file_num][0]
            else:
                return self.raw_edges[file_num][1]
        else:
            try:
                ind, ax = self.select_var(label)
                axis = file['AXIS'][ax][:]
                NX1 = file.attrs['NX'][ind]
                return (axis[1] - axis[0]) * np.arange(NX1) / float(NX1) + axis[0]
            except:
                h5_data = read_hdf(file.filename)
                for ii in range(size(h5_data.axes)):
                    if label == h5_data.axes[ii].attributes['NAME']:
                        break
                axis = [h5_data.axes[ii].axis_min, h5_data.axes[ii].axis_max]
                NX1 = h5_data.shape[ii]
                return (axis[1] - axis[0]) * np.arange(NX1) / float(NX1) + axis[0]

    def axis_bounds(self, file, label):
        ## assuming osiris notation
        ind, ax = self.select_var(label)
        axis = file['AXIS'][ax][:]
        NX1 = file.attrs['NX'][ind]
        return axis

    def axis_label(self, file, label, keyword=None):
        ## assuming osiris notation
        if (keyword == None):
            ind, ax = self.select_var(label)
            try:
                data = file['AXIS'][ax]
            except:
                return label
        else:
            if (keyword == 'r'):
                return self.axis_label(file, 'x2', 'x2')
            data = file[keyword]
        UNITS = data.attrs['UNITS'][0]
        NAME = data.attrs['LONG_NAME'][0]
        return r'${}\/[{}]$'.format(NAME, UNITS)

    def select_var(self, label):
        ind, ax = None, None
        if (label == 'x1'):
            return 0, 'AXIS1'
        elif (label == 'x2'):
            return 1, 'AXIS2'
        else:
            return 2, 'AXIS3'


if __name__ == "__main__":
    main()
