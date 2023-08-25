import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from util.leica import location, Dimensions

matplotlib.use('tkagg')


def make_plot(mlist: np.ndarray[int], dims: Dimensions, directory: str):
    imx = 1314.09 / 1000
    imy = 875.89 / 1000  # mm
    parr = []
    plt.figure(figsize=(18, 18))

    width, height = dims

    for m in mlist:
        x, y = location(m, dims)
        plt.scatter(x * imx, y * imy)
        plt.text(x * imx, y * imy + .03, m, fontsize=9)
        parr.append([m, round(x * imx, 1), round(y * imy, 1)])

    bound_x = [0, (width - 1) * imx, (width - 1) * imx, 0, 0]
    bound_y = [0, 0, (height - 1) * imy, (height - 1) * imy, 0]

    plt.plot(bound_x, bound_y)
    plt.gca().invert_yaxis()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(color='green', linestyle='--', linewidth=0.5)
    plt.savefig(directory + "coordmap.jpg")
    plt.close()
