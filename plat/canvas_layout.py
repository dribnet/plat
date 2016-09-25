import numpy as np
from plat.interpolate import get_interpfn

def create_mine_canvas(rows, cols, y, x, anchors, spherical=True, gaussian=False):
    lerpv = get_interpfn(spherical, gaussian)
    x = np.clip(x, 0, 1)
    y = np.clip(y, 0, 1)

    _, dim = anchors.shape
    u_list = np.zeros((rows, cols, dim))
    # compute anchors
    cur_anchor = 0
    for ys in range(rows):
        for xs in range(cols):
            if anchors is not None and cur_anchor < len(anchors):
                u_list[ys,xs,:] = anchors[cur_anchor]
                cur_anchor = cur_anchor + 1
            else:
                u_list[ys,xs,:] = np.random.normal(0,1, (1, dim))

    # compute coords
    spaceX = 1.0 / (cols - 1)
    scaledX = x * (cols - 1)
    intX = int(scaledX)
    nextX = intX + 1
    fracX = scaledX - intX

    spaceY = 1.0 / (rows - 1)
    scaledY = y * (rows - 1)
    intY = int(scaledY)
    nextY = intY + 1
    fracY = scaledY - intY

    # hack to allow x=1.0 and y=1.0
    if fracX == 0:
        nextX = intX
    if fracY == 0:
        nextY = intY

    h1 = lerpv(fracX, u_list[intY, intX, :], u_list[intY, nextX, :])
    h2 = lerpv(fracX, u_list[nextY, intX, :], u_list[nextY, nextX, :])

    # interpolate vertically
    result = lerpv(fracY, h1, h2)
    if np.isnan(result[0]):
        print("NAN FOUND")
        print("h1: ", h1)
        print("h2: ", h2)
        print("fracx,y", fracX, fracY)
        print("xvars", spaceX, scaledX, intX, nextX, fracX)
        print("yvars", spaceY, scaledY, intY, nextY, fracY)
        print("inputs", x, y, rows, cols)
    return result

