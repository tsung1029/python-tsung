import numpy as np
# import h5_utilities as h5u


def __apply_mask(arr, ax):
    dim = np.ndim(arr)
    # TODO: maybe there are better indexing methods
    if dim == 1:
        for axi in ax[0]:
            arr[axi[0]:axi[1]] = 0
    elif dim == 2:
        for ayi in ax[0]:
            for axi in ax[1]:
                arr[ayi[0]:ayi[1], axi[0]:axi[1]] = 0
    elif dim == 3:
        for azi in ax[0]:
            for ayi in ax[1]:
                for axi in ax[2]:
                    arr[azi[0]:azi[1], ayi[0]:ayi[1], axi[0]:axi[1]] = 0
    return arr


def mask(arr, axes=None, region=None, symmetric=True):
    """
    set values inside region to zeros
    :param arr: array-like data (up to 4-D)
    :param axes: list of axes (data_basic_axis objects)
    :param region: list of triple elements [('x1',l1,u1),...,('xn',ln,un)], where 'xn' (string,) is the direction,
                   ln and un (numbers) are lower and upper limits of the region
    :param symmetric: if true then the mask will also apply to center-symetric regionss
    :return: array with the same dimensions as the input
    """
    # do nothing if region is None
    if not region:
        return arr
    dim = np.ndim(arr)
    sz = np.array(np.shape(arr))
    ax = [[], [], [], []]
    for ci in region:
        # arr is in Fortran ordering: arr[z,y,x] for example
        di = int(ci[0][1]) - 1
        lu = ci[1:3]
        # use axes to convert coordinates to array indices
        if axes:
            lu = np.round((ci[1:3] - axes[di].axis_min) / axes[di].increment).astype(int)
            # check array boundaries
            if lu[0] < 0:
                lu[0] = 0
            if lu[1] > sz[dim - di - 1]:
                lu[1] = sz[dim - di - 1]
        ax[dim - di - 1].append(lu[:])
    # set default region for directions not set
    for i in range(dim):
        if not ax[i]:
            ax[i].append([0, sz[i]])
    arr = __apply_mask(arr, ax)
    if symmetric:
        # the region is roughly symmetric
        for i in range(dim):
            for j, ri in enumerate(ax[i]):
                ax[i][j] = sz[i] - ri[1], sz[i] - ri[0]
    arr = __apply_mask(arr, ax)
    return arr


if __name__ == '__main__':
    data = np.ones((6, 8))
    print data
    da = mask(data, region=[('x2', 3, 5), ('x1', 1, 3)])  # ('x2', 0, 1),
    print "applying mask...\n", data
