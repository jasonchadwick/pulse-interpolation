import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from qctrl_optimize import interpolate

def plot_interpolation(test_points, cs, ref_points, simplices, savepath=None, title=None, draw_simplices=True, aspect=(1,1,1), hist_ylim=None):
    """
    Plot a 3D space with a point for each test point, colored by infidelity.
    """

    fig = plt.figure(figsize=(6,4))
    ax0 = fig.add_subplot(projection='3d')

    xs = test_points[:,0]
    ys = test_points[:,1]
    zs = test_points[:,2]

    colors = ax0.scatter(xs, ys, zs, c=cs, alpha=0.5, norm=matplotlib.colors.LogNorm(1e-5,1))
    ax0.set_box_aspect(aspect)
    ax0.view_init(elev=30, azim=-70)
    cbar = plt.colorbar(colors, label='Infidelity')
    cbar.set_alpha(1)
    cbar.draw_all()
    if draw_simplices:
        for i,(tx,ty,tz) in enumerate(ref_points):
            for pt in interpolate.neighboring_vertices(simplices, i):
                if pt > i:
                    tx1,ty1,tz1 = ref_points[pt]
                    plt.plot([tx,tx1], [ty,ty1], [tz,tz1], c='black', alpha=0.3)
    ax0.grid(False)
    plt.title(title)
    ax0.set_xticks([])
    ax0.set_yticks([])
    ax0.set_zticks([])
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath + 'points.svg')
        plt.savefig(savepath + 'points.pdf')
    plt.show()

    cm = plt.cm.get_cmap("viridis")
    ax = plt.gca()
    _, bins, patches = ax.hist(cs, bins=10**np.arange(-5,1,0.2))
    bin_centers = 0.5*(bins[:-1]+bins[1:])
    norm = matplotlib.colors.LogNorm(1e-5,1)
    for c, p in zip(bin_centers, patches):
        plt.setp(p, "facecolor", cm(norm(c)))
    plt.xscale('log')
    plt.ylim(hist_ylim)
    plt.title('Distribution of test points')
    plt.ylabel('Number of points')
    plt.xlabel('Infidelity')
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath + 'hist.svg')
        plt.savefig(savepath + 'hist.pdf')
    plt.show()

def plot_interpolation_2d(test_points, cs, ref_points, simplices, savepath=None, title=None, aspect=(1,1,1)):
    """
    Plot a 2D space with a point for each test point, colored by infidelity.
    """

    fig,ax0 = plt.subplots()

    xs = test_points[:,0]
    ys = test_points[:,1]

    colors = ax0.scatter(xs, ys, c=cs, vmin=-10,vmax=0, alpha=0.5)
    cbar = plt.colorbar(colors, label=r'$\log_{10}($infidelity$)$')
    cbar.set_alpha(1)
    cbar.draw_all()
    for i,(tx,ty) in enumerate(ref_points):
        for pt in interpolate.neighboring_vertices(simplices, i):
            if pt > i:
                tx1,ty1 = ref_points[pt]
                plt.plot([tx,tx1], [ty,ty1], c='black', alpha=0.3)
    ax0.grid(False)
    plt.title(title)
    ax0.set_xlabel(r't$_x$')
    ax0.set_ylabel(r't$_y$')
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath + 'points.svg')
        plt.savefig(savepath + 'points.pdf')
    plt.show()

    plt.hist(cs, bins=range(-10,1))
    plt.title('Distribution of test points')
    plt.ylabel('Number of points')
    plt.xlabel(r'$\log_{10}($infidelity$)$')
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath + 'hist.svg')
        plt.savefig(savepath + 'hist.pdf')
    plt.show()