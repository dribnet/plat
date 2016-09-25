import numpy as np
from PIL import Image
from plat.interpolate import get_interpfn
from scipy.special import ndtri, ndtr

def grid2img(arr, rows, cols, with_space):
    """Convert an image grid to a single image"""
    N = len(arr)
    channels, height, width = arr[0].shape

    total_height = rows * height
    total_width  = cols * width

    if with_space:
        total_height = total_height + (rows - 1)
        total_width  = total_width + (cols - 1)

    I = np.zeros((channels, total_height, total_width))
    I.fill(1)

    for i in xrange(rows*cols):
        if i < N:
            r = i // cols
            c = i % cols

            cur_im = arr[i]

            if cur_im is not None:
                if with_space:
                    offset_y, offset_x = r*height+r, c*width+c
                else:
                    offset_y, offset_x = r*height, c*width
                I[0:channels, offset_y:(offset_y+height), offset_x:(offset_x+width)] = cur_im

    if(channels == 1):
        out = I.reshape( (total_height, total_width) )
    else:
        out = np.dstack(I)

    out = (255 * out).astype(np.uint8)
    return Image.fromarray(out)

def create_chain_grid(rows, cols, dim, space, anchors, spherical, gaussian):
    """Create a grid of latents with chained-analogy layout"""
    ## Map (r/s + c/w - 1) anchors to r/s * c/s, then call create_mine_grid
    num_row_anchors = (rows + space - 1) / space
    num_col_anchors = (cols + space - 1) / space
    u_list = np.zeros((num_row_anchors, num_col_anchors, dim))    
    for y in range(num_row_anchors):
        for x in range(num_col_anchors):
            if x == 0 or y == 0:
                if x == 0 and y == 0:
                    anchor_index = 0
                elif x == 0:
                    anchor_index = y * 2 - 1
                else:
                    anchor_index = x * 2
                u_list[y,x,:] = anchors[anchor_index]
            else:
                anal_vec = u_list[y,x-1,:] + (u_list[y-1,x,:] - u_list[y-1,x-1,:])
                anal_len = np.linalg.norm(anal_vec)
                anal_unit_vec = np.nan_to_num(anal_vec / anal_len)
                avg_len = (np.linalg.norm(u_list[y,x-1,:]) +
                    np.linalg.norm(u_list[y-1,x,:]) +
                    np.linalg.norm(u_list[y-1,x-1,:])) / 3.0
                u_list[y,x,:] = avg_len * anal_unit_vec

    u_grid = u_list.reshape(num_row_anchors * num_col_anchors, dim)
    return create_mine_grid(rows, cols, dim, space, u_grid, spherical, gaussian)

def create_mine_grid(rows, cols, dim, space, anchors, spherical, gaussian):
    """Create a grid of latents with splash layout"""
    lerpv = get_interpfn(spherical, gaussian)

    u_list = np.zeros((rows, cols, dim))
    # compute anchors
    cur_anchor = 0
    for y in range(rows):
        for x in range(cols):
            if y%space == 0 and x%space == 0:
                if anchors is not None and cur_anchor < len(anchors):
                    u_list[y,x,:] = anchors[cur_anchor]
                    cur_anchor = cur_anchor + 1
                else:
                    u_list[y,x,:] = np.random.normal(0,1, (1, dim))
    # interpolate horizontally
    for y in range(rows):
        for x in range(cols):
            if y%space == 0 and x%space != 0:
                lastX = space * (x // space)
                nextX = lastX + space
                fracX = (x - lastX) / float(space)
#                 print("{} - {} - {}".format(lastX, nextX, fracX))
                u_list[y,x,:] = lerpv(fracX, u_list[y, lastX, :], u_list[y, nextX, :])
    # interpolate vertically
    for y in range(rows):
        for x in range(cols):
            if y%space != 0:
                lastY = space * (y // space)
                nextY = lastY + space
                fracY = (y - lastY) / float(space)
                u_list[y,x,:] = lerpv(fracY, u_list[lastY, x, :], u_list[nextY, x, :])

    u_grid = u_list.reshape(rows * cols, dim)

    return u_grid

def create_gradient_grid(rows, cols, dim, analogy, anchors, spherical, gaussian):
    """Create a grid of latents with gradient layout (includes analogy)"""
    lerpv = get_interpfn(spherical, gaussian)
    hyper = False

    numsamples = rows * cols
    u_list = np.zeros((numsamples, dim))
    if anchors is not None:
        # xmin_ymin, xmax_ymin, xmin_ymax = anchors[0:3]
        xmin_ymin, xmin_ymax, xmax_ymin = anchors[0:3]
    else:
        xmin_ymin = np.random.normal(0, 1, dim)
        xmax_ymin = np.random.normal(0, 1, dim)
        xmin_ymax = np.random.normal(0, 1, dim)
    if(analogy):
        xmax_ymax = xmin_ymax + (xmax_ymin - xmin_ymin)
        if hyper:
            tl = xmin_ymin
            tr = xmax_ymin
            bl = xmin_ymax
            xmax_ymax = bl + (tr - tl)
            xmin_ymax = bl - (tr - tl)
            xmax_ymin = tr + (tl - bl)
            xmin_ymin = xmin_ymax + (xmax_ymin - xmax_ymax)
    elif anchors is not None:
        xmax_ymax = anchors[3]
    else:
        xmax_ymax = np.random.normal(0, 1, dim)

    for y in range(rows):
        if  y == 0:
            # allows rows == 0
            y_frac = 0
        else:
            y_frac = y / (rows - 1.0)
        xmin_ycur = lerpv(y_frac, xmin_ymin, xmin_ymax)
        xmax_ycur = lerpv(y_frac, xmax_ymin, xmax_ymax)
        for x in range(cols):
            if x == 0:
                # allows cols == 0
                x_frac = 0
            else:
                x_frac = x / (cols - 1.0)
            xcur_ycur = lerpv(x_frac, xmin_ycur, xmax_ycur)
            n = y * cols + x
            u_list[n:n+1,:] = xcur_ycur

    return u_list

def create_fan_grid(z_dim, cols, rows, gaussian_prior=True, interleaves=0, shuffles=0):
    """This is a legacy grid layout"""
    sqrt2 = 1.0

    def lerpIt(val, disp, low, high):
        sumval = ((val + disp + sqrt2) / (2 * sqrt2)) % 2.0
        if sumval < 0:
            print("AHHH ERROR")
            sumval = -sumval
        if sumval < 1:
            zeroToOne = sumval
        else:
            zeroToOne = 2.0 - sumval
        return low + (high - low) * zeroToOne

    def lerpTo(val, low, high):
        zeroToOne = np.clip((val + sqrt2) / (2 * sqrt2), 0, 1)
        return low + (high - low) * zeroToOne

    def lerp(val, low, high):
        return low + (high - low) * val

    def pol2cart(phi):
        x = np.cos(phi)
        y = np.sin(phi)
        return(x, y)

    #  http://stackoverflow.com/a/5347492
    # >>> interleave(np.array(range(6)))
    # array([0, 3, 1, 4, 2, 5])
    def interleave(offsets):
        shape = offsets.shape
        split_point = int(shape[0] / 2)
        a = np.array(offsets[:split_point])
        b = np.array(offsets[split_point:])
        c = np.empty(shape, dtype=a.dtype)
        c[0::2] = a
        c[1::2] = b
        return c

    def shuffle(offsets):
        np.random.shuffle(offsets)

    offsets = []
    displacements = []
    for i in range(z_dim):
        offsets.append(pol2cart(i * np.pi / z_dim))
        displacements.append(1.0 * i / z_dim)
    offsets = np.array(offsets)
    displacements = np.array(displacements)

    for i in range(interleaves):
        offsets = interleave(offsets)

    for i in range(shuffles):
        shuffle(offsets)

    ul = []
    # range_high = 0.95
    # range_low = 1 - range_high
    range_high = 0.997  # 3 standard deviations
    range_low = 1 - range_high
    for r in range(rows):
        # xf = lerp(r / (rows-1.0), -1.0, 1.0)
        xf = (r - (rows / 2.0) + 0.5) / ((rows-1) / 2.0 + 0.5)
        for c in range(cols):
            # yf = lerp(c / (cols-1.0), -1.0, 1.0)
            yf = (c - (cols / 2.0) + 0.5) / ((cols-1) / 2.0 + 0.5)
            coords = map(lambda o: np.dot([xf, yf], o), offsets)
            ranged = map(lambda n, d:lerpIt(n, d, range_low, range_high), coords, displacements)
            # ranged = map(lambda n:lerpTo(n, range_low, range_high), coords)
            if(gaussian_prior):
                cdfed = map(ndtri, ranged)
            else:
                cdfed = ranged
            ul.append(cdfed)
    u = np.array(ul).reshape(rows,cols,z_dim).astype('float32')
    return u
