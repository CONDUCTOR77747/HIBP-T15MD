'''
Heavy Ion Beam Probe graphic library
'''
# %% imports

import os
import wire
import pylab
import numpy as np
from itertools import cycle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.stats import gaussian_kde
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import hibplib as hb
import hibpcalc.fields as fields
from oscillation import CoordConverterDShapeSimple

try:
    import visvis as vv
except ModuleNotFoundError:
    print('module visvis NOT FOUND')
    pass

try:
    import alphashape
except ModuleNotFoundError:
    print('module alphashape NOT FOUND')
    pass

# %% Magnetic field plots


def plot_3d(B, wires, volume_corner1, volume_corner2,
            grid, resolution, cutoff=2):
    '''
    plot absolute values of B in 3d with visvis
    B : magnetic field values array (has 3 dimensions) [T]
    wires : list of wire objects
    '''

    Babs = np.linalg.norm(B, axis=1)
    Babs[Babs > cutoff] = np.nan

    # draw results
    # prepare axes
    a = vv.gca()
    a.cameraType = '3d'
    a.daspectAuto = False

    vol = Babs.reshape(grid.shape[1:]).T
    vol = vv.Aarray(vol, sampling=(resolution, resolution, resolution),
                    origin=(volume_corner1[2], volume_corner1[1],
                            volume_corner1[0]))
    # set labels
    vv.xlabel('x axis')
    vv.ylabel('y axis')
    vv.zlabel('z axis')

    wire.vv_PlotWires(wires)

    vv.volshow2(vol, renderStyle='mip', cm=vv.CM_JET)
    vv.colorbar()
    app = vv.use()
    app.Run()


# %% matplotlib plot 2D magnetic field
def plot_2d(B, points, plane='xy', cutoff=2, n_contours=50):
    '''
    make contour plot of B in XZ or XY plane
    B : magnetic field values array (has 3 dimensions) [T]
    points : coordinates for points for B vectors to start on
    '''

    pf_coils = hb.import_PFcoils('PFCoils.dat')
    # 2d quiver
    # get 2D values from one plane with Y = 0
    fig = plt.figure()
    ax = fig.gca()
    if plane == 'xz':
        # choose y position
        mask = (np.around(points[:, 1], 3) == 0.1)
        B = B[mask]
        Babs = np.linalg.norm(B, axis=1)
        B[Babs > cutoff] = [np.nan, np.nan, np.nan]
        points = points[mask]
#        ax.quiver(points[:, 0], points[:, 2], B[:, 0], B[:, 2], scale=2.0)

        X = np.unique(points[:, 0])
        Z = np.unique(points[:, 2])
        cs = ax.contour(X, Z, Babs.reshape([len(X), len(Z)]).T, n_contours)
        ax.clabel(cs)
        plt.xlabel('x')
        plt.ylabel('z')

    elif plane == 'xy':
        # get coil inner and outer profile
        TF_coil_filename = 'TFCoil.dat'
        # load coil contours from txt
        TF_coil = np.loadtxt(TF_coil_filename) / 1000  # [m]

        # plot toroidal coil
        ax.plot(TF_coil[:, 0], TF_coil[:, 1], '--', color='k')
        ax.plot(TF_coil[:, 2], TF_coil[:, 3], '--', color='k')

        # plot pf coils
        for coil in pf_coils.keys():
            xc = pf_coils[coil][0]
            yc = pf_coils[coil][1]
            dx = pf_coils[coil][2]
            dy = pf_coils[coil][3]
            ax.add_patch(Rectangle((xc-dx/2, yc-dy/2), dx, dy,
                                   linewidth=1, edgecolor='r', facecolor='r'))

        # choose z position
        mask = (np.around(points[:, 2], 3) == 0.0)
        B = B[mask]
        Babs = np.linalg.norm(B, axis=1)
        B[Babs > cutoff] = [np.nan, np.nan, np.nan]
        points = points[mask]
#        ax.quiver(points[:, 0], points[:, 1], B[:, 0], B[:, 1], scale=20.0)

        X = np.unique(points[:, 0])
        Y = np.unique(points[:, 1])
        cs = ax.contour(X, Y, Babs.reshape([len(X), len(Y)]).T, n_contours)
        ax.clabel(cs)

        plt.xlabel('x')
        plt.ylabel('y')

    plt.axis('equal')

    clb = plt.colorbar(cs)
    clb.set_label('V', labelpad=-40, y=1.05, rotation=0)

    plt.show()


# %% stream plot of magnetic field
def plot_B_stream(B, volume_corner1, volume_corner2, resolution,
                  grid, color='r', dens=1.0, plot_sep=True):
    '''
    stream plot of magnetic field
    '''

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    set_axes_param(ax1, 'X (m)', 'Y (m)')
    set_axes_param(ax2, 'X (m)', 'Z (m)')

    plot_geometry(ax1, plot_sep=plot_sep)

    x = np.arange(volume_corner1[0], volume_corner2[0], resolution)
    y = np.arange(volume_corner1[1], volume_corner2[1], resolution)
    z = np.arange(volume_corner1[2], volume_corner2[2], resolution)

    Bx = B[:, 0].reshape(grid.shape[1:])
    By = B[:, 1].reshape(grid.shape[1:])
    Bz = B[:, 2].reshape(grid.shape[1:])

    # choose z position
    z_cut = np.where(abs(z) < 0.001)[0][0]  # Bx.shape[2]//2
    # choose y position
    y_cut = np.where(abs(y) < 0.001)[0][0]  # Bx.shape[1]//2

    ax1.streamplot(x, y, Bx[:, :, z_cut].swapaxes(0, 1),
                   By[:, :, z_cut].swapaxes(0, 1), color=color, density=dens)
    ax2.streamplot(x, z, Bx[:, y_cut, :].swapaxes(0, 1),
                   Bz[:, y_cut, :].swapaxes(0, 1), color=color, density=dens)
    plt.show()


# %% matplotlib plot 3D
def plot_3dm(B, wires, points, cutoff=2):
    '''
    plot 3d quiver of B using matplotlib
    '''
    Babs = np.linalg.norm(B, axis=1)
    B[Babs > cutoff] = [np.nan, np.nan, np.nan]

    fig = plt.figure()
    # 3d quiver
    ax = fig.gca(projection='3d')
    wire.mpl3d_PlotWires(wires, ax)
#    ax.quiver(points[:, 0], points[:, 1], points[:, 2],
#              B[:, 0], B[:, 1], B[:, 2], length=20)
    plt.show()


# %% Electric field plots
def plot_contours(X, Y, Z, U, upper_plate_flag, lower_plate_flag,
                  n_contours=30, plates_color='k'):
    '''
    contour plot of potential U
    X, Y, Z : mesh ranges in X, Y and Z respectively [m]
    U :  plate's voltage [V]
    n_contours :  number of contour lines
    '''

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharex=True)
    set_axes_param(ax1, 'X (m)', 'Y (m)')
    set_axes_param(ax2, 'Z (m)', 'Y (m)')

    z_cut = U.shape[2]//2
    CS = ax1.contour(X, Y, U[:, :, z_cut].swapaxes(0, 1), n_contours)
    # add the edge of the domain
    domain = patches.Rectangle((min(X), min(Y)), max(X)-min(X), max(Y)-min(Y),
                               linewidth=2, linestyle='--', edgecolor='k',
                               facecolor='none')
    ax1.add_patch(domain)

    x_cut = U.shape[0]//2
    ax2.contour(Z, Y, U[x_cut, :, :], n_contours)
    # add the edge of the domain
    domain = patches.Rectangle((min(Z), min(Y)), max(Z)-min(Z), max(Y)-min(Y),
                               linewidth=2, linestyle='--', edgecolor='k',
                               facecolor='none')
    ax2.add_patch(domain)

    # add plates
    x, y, z = np.meshgrid(X, Y, Z, indexing='ij')

    ax1.plot(x[:, :, z_cut][upper_plate_flag[:, :, z_cut]],
             y[:, :, z_cut][upper_plate_flag[:, :, z_cut]], 'o', color='k')
    ax1.plot(x[:, :, z_cut][lower_plate_flag[:, :, z_cut]],
             y[:, :, z_cut][lower_plate_flag[:, :, z_cut]], 'o', color='k')

    ax2.plot(z[x_cut, :, :][upper_plate_flag[x_cut, :, :]],
             y[x_cut, :, :][upper_plate_flag[x_cut, :, :]], 'o', color='k')
    ax2.plot(z[x_cut, :, :][lower_plate_flag[x_cut, :, :]],
             y[x_cut, :, :][lower_plate_flag[x_cut, :, :]], 'o', color='k')

    # clb = plt.colorbar(CS)
    # clb.set_label('V', labelpad=-40, y=1.05, rotation=0)
    plt.show()


# %%
def plot_contours_zy(X, Y, Z, U, upper_plate_flag, lower_plate_flag,
                     n_contours=30, plates_color='k'):

    fig, ax1 = plt.subplots()
    set_axes_param(ax1, 'Z (m)', 'Y (m)')

    x_cut = U.shape[0]//2
    ax1.contour(Z, Y, U[x_cut, :, :], n_contours)
    # add the edge of the domain
    domain = patches.Rectangle((min(Z), min(Y)), max(Z)-min(Z), max(Y)-min(Y),
                               linewidth=2, linestyle='--', edgecolor='k',
                               facecolor='none')
    ax1.add_patch(domain)

    x, y, z = np.meshgrid(X, Y, Z, indexing='ij')
    ax1.plot(z[x_cut, :, :][upper_plate_flag[x_cut, :, :]],
             y[x_cut, :, :][upper_plate_flag[x_cut, :, :]], 'o', color='r')
    ax1.plot(z[x_cut, :, :][lower_plate_flag[x_cut, :, :]],
             y[x_cut, :, :][lower_plate_flag[x_cut, :, :]], 'o', color='k')

    # clb = plt.colorbar(CS)
    # clb.set_label('V', labelpad=-40, y=1.05, rotation=0)
    plt.show()


# %%
def plot_contours_xz(X, Y, Z, U, upper_plate_flag, lower_plate_flag,
                     n_contours=30, plates_color='k'):

    fig, ax1 = plt.subplots()
    set_axes_param(ax1, 'X (m)', 'Z (m)')

    y_cut = U.shape[1]//2
    ax1.contour(X, Z, U[:, y_cut, :].swapaxes(0, 1), n_contours)
    # add the edge of the domain
    domain = patches.Rectangle((min(X), min(Z)), max(X)-min(X), max(Z)-min(Z),
                               linewidth=2, linestyle='--', edgecolor='k',
                               facecolor='none')
    ax1.add_patch(domain)

    x, y, z = np.meshgrid(X, Y, Z, indexing='ij')
    ax1.plot(x[:, y_cut, :][upper_plate_flag[:, y_cut, :]],
             z[:, y_cut, :][upper_plate_flag[:, y_cut, :]], 'o', color='k')
    ax1.plot(x[:, y_cut, :][lower_plate_flag[:, y_cut, :]],
             z[:, y_cut, :][lower_plate_flag[:, y_cut, :]], 'o', color='k')
    plt.show()


# %%
def plot_stream_zy(X, Y, Z, Ex, Ey, Ez, upper_plate_flag, lower_plate_flag,
                   dens=1.0, plates_color='k'):

    fig, ax1 = plt.subplots()
    set_axes_param(ax1, 'Z (m)', 'Y (m)')

    x_cut = Ey.shape[0]//2
    ax1.streamplot(Z, Y, Ez[x_cut, :, :], Ey[x_cut, :, :], density=dens)
    # add the edge of the domain
    domain = patches.Rectangle((min(Z), min(Y)), max(Z)-min(Z), max(Y)-min(Y),
                               linewidth=2, linestyle='--', edgecolor='k',
                               facecolor='none')
    ax1.add_patch(domain)

    x, y, z = np.meshgrid(X, Y, Z, indexing='ij')
    ax1.plot(z[x_cut, :, :][upper_plate_flag[x_cut, :, :]],
             y[x_cut, :, :][upper_plate_flag[x_cut, :, :]], 'o', color='r')
    ax1.plot(z[x_cut, :, :][lower_plate_flag[x_cut, :, :]],
             y[x_cut, :, :][lower_plate_flag[x_cut, :, :]], 'o', color='k')
    plt.show()


# %%
def plot_stream(X, Y, Z, Ex, Ey, Ez, upper_plate_flag, lower_plate_flag,
                dens=1.0, plates_color='k'):
    '''
    stream plot of Electric field in xy, xz, zy planes
    X, Y, Z : mesh ranges in X, Y and Z respectively [m]
    Ex, Ey, Ez : Electric field components [V/m]
    '''
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharex=True)
    set_axes_param(ax1, 'X (m)', 'Y (m)')
    set_axes_param(ax2, 'Z (m)', 'Y (m)')

    z_cut = Ex.shape[2]//2  # z position of XY cut

    ax1.streamplot(X, Y, Ex[:, :, z_cut].swapaxes(0, 1),
                   Ey[:, :, z_cut].swapaxes(0, 1), density=dens)
    # add the edge of the domain
    domain = patches.Rectangle((min(X), min(Y)), max(X)-min(X), max(Y)-min(Y),
                               linewidth=2, linestyle='--', edgecolor='k',
                               facecolor='none')
    ax1.add_patch(domain)

    x_cut = Ey.shape[0]//2  # x position of ZY cut
    ax2.streamplot(Z, Y, Ez[x_cut, :, :], Ey[x_cut, :, :], density=dens)
    # add the edge of the domain
    domain = patches.Rectangle((min(Z), min(Y)), max(Z)-min(Z), max(Y)-min(Y),
                               linewidth=2, linestyle='--', edgecolor='k',
                               facecolor='none')
    ax2.add_patch(domain)

    # add plates
    x, y, z = np.meshgrid(X, Y, Z, indexing='ij')

    ax1.plot(x[:, :, z_cut][upper_plate_flag[:, :, z_cut]],
             y[:, :, z_cut][upper_plate_flag[:, :, z_cut]], 'o', color='k')
    ax1.plot(x[:, :, z_cut][lower_plate_flag[:, :, z_cut]],
             y[:, :, z_cut][lower_plate_flag[:, :, z_cut]], 'o', color='k')

    ax2.plot(z[x_cut, :, :][upper_plate_flag[x_cut, :, :]],
             y[x_cut, :, :][upper_plate_flag[x_cut, :, :]], 'o', color='k')
    ax2.plot(z[x_cut, :, :][lower_plate_flag[x_cut, :, :]],
             y[x_cut, :, :][lower_plate_flag[x_cut, :, :]], 'o', color='k')
    plt.show()


# %%
def plot_quiver(X, Y, Z, Ex, Ey, Ez):
    '''
    quiver plot of Electric field in xy, xz, zy planes
    X, Y, Z : mesh ranges in X, Y and Z respectively [m]
    Ex, Ey, Ez : Electric components [V/m]
    '''
#    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    # fig, ax1 = plt.subplots(nrows=1, ncols=1)

    z_cut = Ex.shape[2]//2 + 20  # z position of XY cut

    ax1.quiver(X, Y, Ex[:, :, z_cut].swapaxes(0, 1),
               Ey[:, :, z_cut].swapaxes(0, 1))
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.grid(True)
    ax1.axis('equal')
    # add the edge of the domain
    domain = patches.Rectangle((min(X), min(Y)), max(X)-min(X), max(Y)-min(Y),
                               linewidth=2, linestyle='--', edgecolor='k',
                               facecolor='none')
    ax1.add_patch(domain)

    x_cut = Ey.shape[0]//2  # x position of ZY cut
    ax2.quiver(Z, Y, Ez[x_cut, :, :], Ey[x_cut, :, :])
    ax2.set_xlabel('Z (m)')
    ax2.set_ylabel('Y (m)')
    ax2.grid(True)
    ax2.axis('equal')
    # add the edge of the domain
    domain = patches.Rectangle((min(Z), min(Y)), max(Z)-min(Z), max(Y)-min(Y),
                               linewidth=2, linestyle='--', edgecolor='k',
                               facecolor='none')
    ax2.add_patch(domain)
#
#    y_cut = Ex.shape[1]//2 # y position of XZ cut
#    ax3.quiver(X, Z, Ex[:, y_cut, :].swapaxes(0, 1),
#                     Ez[:, y_cut, :].swapaxes(0, 1))
#    ax3.set_xlabel('X (m)')
#    ax3.set_ylabel('Z (m)')
#    ax3.grid(True)
#    ax3.axis('equal')
#    # add the edge of the domain
#    domain = patches.Rectangle((min(X), min(Z)), max(X)-min(X), max(Z)-min(Z),
#                               linewidth=2, linestyle='--', edgecolor='k',
#                               facecolor='none')
#    ax3.add_patch(domain)
    plt.show()


# %%
def plot_quiver3d(X, Y, Z, Ex, Ey, Ez, UP_rotated, LP_rotated, n_skip=5):
    '''
    3d quiver plot of Electric field
    X, Y, Z : mesh ranges in X, Y and Z respectively
    Ex, Ey, Ez :  plate's U gradient components
    UP_rotated, LP_rotated : upper's and lower's plate angle coordinates
    n_skip :  number of planes to skip before plotting
    '''
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # plot plates (very scematically)
    ax.plot(UP_rotated[:, 0], UP_rotated[:, 1], UP_rotated[:, 2],
            '-o', color='b')
    ax.plot(LP_rotated[:, 0], LP_rotated[:, 1], LP_rotated[:, 2],
            '-o', color='r')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.grid(True)

    x_pos = X.shape[0]//2
    y_pos = Y.shape[1]//2
#    z_pos = Z.shape[2]//2

    skip = (x_pos, slice(None, None, 3*n_skip), slice(None, None, n_skip))
    ax.quiver3D(X[skip], Y[skip], Z[skip],
                Ex[skip],
                Ey[skip],
                Ez[skip], length=0.01, normalize=True)

    skip = (slice(None, None, 3*n_skip), y_pos, slice(None, None, n_skip))
    ax.quiver3D(X[skip], Y[skip], Z[skip],
                Ex[skip],
                Ey[skip],
                Ez[skip], length=0.01, normalize=True)

    ax.axis('equal')
    plt.show()


# %%
def plot_geometry(ax, TF_coil_filename='TFCoil.dat',
                  camera_data_filename='T15_vessel.txt',
                  separatrix_data_filename='T15_sep.txt',
                  PFCoils_data_filename='PFCoils.dat',
                  major_radius=1.5, plot_sep=True):
    '''
    plot toroidal and poloidal field coils, camera and separatrix in XY plane
    '''
    # load coil contours from txt
    TF_coil = np.loadtxt(TF_coil_filename) / 1000  # [m]

    # plot toroidal coil
    ax.plot(TF_coil[:, 0], TF_coil[:, 1], '--', color='k')
    ax.plot(TF_coil[:, 2], TF_coil[:, 3], '--', color='k')

    # get T-15 camera and plasma contours
    camera = np.loadtxt(camera_data_filename)/1000
    ax.plot(camera[:, 0], camera[:, 1], color='tab:blue')

    # plot first wall
    in_fw = np.loadtxt('infw.txt') / 1000  # [m]
    out_fw = np.loadtxt('outfw.txt') / 1000  # [m]
    ax.plot(in_fw[:, 0], in_fw[:, 1], color='k')
    ax.plot(out_fw[:, 0], out_fw[:, 1], color='k')

    if plot_sep:
        if separatrix_data_filename is not None:
            separatrix = np.loadtxt(separatrix_data_filename)/1000
            ax.plot(separatrix[:, 0] + major_radius, separatrix[:, 1],
                    color='b')  # 'tab:orange')

    if PFCoils_data_filename is not None:
        pf_coils = fields.import_PFcoils(PFCoils_data_filename)

        # plot pf coils
        for coil in pf_coils.keys():
            xc = pf_coils[coil][0]
            yc = pf_coils[coil][1]
            dx = pf_coils[coil][2]
            dy = pf_coils[coil][3]
            ax.add_patch(Rectangle((xc-dx/2, yc-dy/2), dx, dy,
                                   linewidth=1, edgecolor='tab:gray',
                                   facecolor='tab:gray'))
    plt.show()


# %%
def set_axes_param(ax, xlabel, ylabel, isequal=True, fontsize=14):
    '''
    format axes
    '''
    ax.grid(True)
    ax.grid(which='major', color='tab:gray')  # draw primary grid
    ax.minorticks_on()  # make secondary ticks on axes
    # draw secondary grid
    ax.grid(which='minor', color='tab:gray', linestyle=':')
    ax.xaxis.set_tick_params(width=2, labelsize=fontsize)  # increase tick size
    ax.yaxis.set_tick_params(width=2, labelsize=fontsize)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    if isequal:
        ax.axis('equal')

#  plt.rcParams.update({'font.family':'Copperplate Gothic Light'})
#  ax.yaxis.set_tick_params(labelsize=2, labelcolor='b', fontweight='bold', rotation=25, width=2)

# %%
def set_lim(ax, coord, lim):
    xlim = ax.get_xlim()
    dx = xlim[1] - xlim[0]
    x0 = 0.5*(xlim[1] + xlim[0])
    ylim = ax.get_ylim()
    dy = ylim[1] - ylim[0]
    y0 = 0.5*(ylim[1] + ylim[0])

    if coord == 'Y':
        new_ylim = lim
        dy_new = new_ylim[1] - new_ylim[0]
        dx_new = dx*dy_new/dy
        new_xlim = (x0-dx_new*0.5, x0+dx_new*0.5)
    elif coord == 'X':
        new_xlim = lim
        dx_new = new_xlim[1] - new_xlim[0]
        dy_new = dy*dx_new/dx
        new_ylim = (y0-dy_new*0.5, y0+dy_new*0.5)
    else:
        raise Exception("unexcpected coord %s" % coord)

    ax.set_ylim(new_ylim)
    ax.set_xlim(new_xlim)

# %% Plot trajectories


def plot_traj(traj_list, geom, Ebeam, UA2, Btor, Ipl, full_primary=False,
              plot_analyzer=False, subplots_vertical=False, scale=5):
    '''
    plot primary and secondary trajectories
    traj_list : list of Traj objects
    geom : Geometry object
    Ebeam : beam energy [keV]
    UA2 : A2 voltage [kV]
    config : magnetic configuretion
    '''
    if subplots_vertical:
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True,
                                       gridspec_kw={'height_ratios':
                                                    [scale, 1]})
    else:
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

    set_axes_param(ax1, 'X (m)', 'Y (m)')
    set_axes_param(ax2, 'X (m)', 'Z (m)')

    # plot geometry
    geom.plot(ax1, axes='XY', plot_analyzer=plot_analyzer)
    geom.plot(ax2, axes='XZ', plot_analyzer=plot_analyzer)

    # plot trajectory
    for tr in traj_list:
        if tr.Ebeam == Ebeam and tr.U['A2'] == UA2:
            # plot primary
            tr.plot_prim(ax1, axes='XY', color='k', full_primary=full_primary)
            tr.plot_prim(ax2, axes='XZ', color='k', full_primary=full_primary)

            ax1.set_title('E={} keV, UA2={} kV, Btor = {} T, Ipl = {} MA'
                          .format(Ebeam, UA2, Btor, Ipl))

            if plot_analyzer and hasattr(tr, 'RV_sec_slit'):
                n_slits = len(tr.RV_sec_slit)
                for i in range(n_slits):
                    for sec_tr in tr.RV_sec_slit[i]:
                        ax1.plot(sec_tr[:, 0], sec_tr[:, 1], color='r')
                        ax2.plot(sec_tr[:, 0], sec_tr[:, 2], color='r')
                        ax1.plot(sec_tr[0, 0], sec_tr[0, 1], 'o', color='r',
                                 markerfacecolor='w')
                        ax2.plot(sec_tr[0, 0], sec_tr[0, 2], 'o', color='r',
                                 markerfacecolor='w')
            else:
                tr.plot_sec(ax1, axes='XY', color='r')
                tr.plot_sec(ax2, axes='XZ', color='r')

            break
    plt.show()

# %%


def plot_traj_one_subplot(traj_list, geom, Ebeam, UA2, Btor, Ipl, ax=None,
                          axes='XY', fontsize_axes=14, full_primary=False,
                          plot_analyzer=False, scale=5, cut=-1):
    if ax is None:
        plt.figure()
        ax = plt.gca()
        set_axes_param(ax, '%s, м' %
                       axes[0], '%s, м' % axes[1], fontsize=fontsize_axes)

    geom.plot(ax, axes=axes, plot_analyzer=plot_analyzer)
    # plot trajectory
    for tr in traj_list:
        if tr.Ebeam == Ebeam and tr.U['A2'] == UA2:
            # plot primary
            tr.plot_prim(ax, axes=axes, color='k', full_primary=full_primary)

            ax.set_title('E={} keV, UA2={} kV, Btor = {} T, Ipl = {} MA'
                         .format(Ebeam, UA2, Btor, Ipl))

            if plot_analyzer and hasattr(tr, 'RV_sec_slit'):
                n_slits = len(tr.RV_sec_slit)
                for i in range(n_slits):
                    for sec_tr in tr.RV_sec_slit[i]:
                        ax.plot(sec_tr[:cut, 0], sec_tr[:cut, 1], color='r')
                        ax.plot(sec_tr[0, 0], sec_tr[0, 1], 'o', color='r',
                                markerfacecolor='w')
            else:
                # tr.plot_sec(ax, axes=axes, color='r')
                ax.plot(tr.RV_sec[:cut, 0], tr.RV_sec[:cut, 1], color='r')

            break
    plt.show()
# %%


def plot_fan(traj_list, geom, Ebeam, UA2, Btor, Ipl, plot_traj=True,
             plot_last_points=True, plot_analyzer=False, plot_all=False,
             full_primary=True):
    '''
    plot fan of trajectories in xy, xz and zy planes
    '''

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)
    # fig, (ax1, ax3) = plt.subplots(nrows=1, ncols=2)

    set_axes_param(ax1, 'X (m)', 'Y (m)')
    set_axes_param(ax2, 'X (m)', 'Z (m)')
    set_axes_param(ax3, 'Z (m)', 'Y (m)')

    # get color cycler
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    colors = cycle(colors)
    # get marker cycler
    markers = cycle(('o', 'v', '^', '<', '>', '*', 'D', 'P', 'd'))

    # plot geometry
    geom.plot(ax1, axes='XY', plot_analyzer=plot_analyzer)
    geom.plot(ax2, axes='XZ', plot_analyzer=plot_analyzer)
    geom.plot(ax3, axes='ZY', plot_analyzer=plot_analyzer)

    ax1.set_title('E={} keV, Btor={} T, Ipl={} MA'
                  .format(Ebeam, Btor, Ipl))

    # invert Z ax
    # ax3.invert_xaxis()
    sec_color = 'r'
    marker = 'o'

    for tr in traj_list:
        if plot_all:
            UA2 = tr.U['A2']
            Ebeam_new = tr.Ebeam
            if Ebeam != Ebeam_new:
                Ebeam = Ebeam_new
                sec_color = next(colors)
                marker = next(markers)

        if tr.Ebeam == Ebeam and tr.U['A2'] == UA2:
            if plot_traj:
                # plot primary
                tr.plot_prim(ax1, axes='XY', color='k',
                             full_primary=full_primary)
                tr.plot_prim(ax2, axes='XZ', color='k',
                             full_primary=full_primary)
                tr.plot_prim(ax3, axes='ZY', color='k',
                             full_primary=full_primary)
                # plot fan of secondaries
                tr.plot_fan(ax1, axes='XY', color=sec_color)
                tr.plot_fan(ax2, axes='XZ', color=sec_color)
                tr.plot_fan(ax3, axes='ZY', color=sec_color)

            if plot_last_points:
                last_points = []
                for _traj_points in tr.Fan:
                    last_points.append(_traj_points[-1, :])

                last_points = np.array(last_points)
                if last_points.shape == (0,):
                    print('last_points is empty')
                else:
                    ax1.plot(last_points[:, 0], last_points[:, 1],
                             '--', c=sec_color)
                    ax1.scatter(last_points[:, 0], last_points[:, 1],
                                marker=marker, c='w', edgecolors=sec_color,
                                label='E={:.1f}, UA2={:.1f}'.format(Ebeam, UA2))
                    ax2.plot(last_points[:, 0], last_points[:, 2],
                             '--o', c=sec_color, mfc='w', mec=sec_color)
                    ax3.plot(last_points[:, 2], last_points[:, 1],
                             '--', c=sec_color)
                    ax3.scatter(last_points[:, 2], last_points[:, 1],
                                marker=marker, c='w', edgecolors=sec_color)

            if not plot_all:
                ax1.set_title('E={} keV, UA2={} kV, UB2={:.1f} kV, '
                              'Btor={} T, Ipl={} MA'
                              .format(Ebeam, UA2, tr.U['B2'], Btor, Ipl))
                break
    ax1.legend()
    plt.show()


# %%
def plot_scan(traj_list, geom, Ebeam, Btor, Ipl, full_primary=False,
              plot_analyzer=False, plot_det_line=False,
              subplots_vertical=False, scale=5, color_sec='r'):
    '''
    plot scan for a particular energy Ebeam in XY and XZ planes
    '''
    if subplots_vertical:
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True,
                                       gridspec_kw={'height_ratios':
                                                    [scale, 1]})
    else:
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

    set_axes_param(ax1, 'X (m)', 'Y (m)')
    set_axes_param(ax2, 'X (m)', 'Z (m)')

    # plot geometry
    geom.plot(ax1, axes='XY', plot_analyzer=plot_analyzer)
    geom.plot(ax2, axes='XZ', plot_analyzer=plot_analyzer)

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    A2list = []
    det_line = np.empty([0, 3])

    for tr in traj_list:
        if tr.Ebeam == Ebeam:
            A2list.append(tr.U['A2'])
            if plot_det_line:
                det_line = np.vstack([det_line, tr.RV_sec[0, 0:3]])
            # plot primary
            tr.plot_prim(ax1, axes='XY', color='k', full_primary=full_primary)
            tr.plot_prim(ax2, axes='XZ', color='k', full_primary=full_primary)
            # plot secondary
            if plot_analyzer and hasattr(tr, 'RV_sec_slit'):
                n_slits = len(tr.RV_sec_slit)
                for i in range(n_slits):
                    for sec_tr in tr.RV_sec_slit[i]:
                        ax1.plot(sec_tr[:, 0], sec_tr[:, 1], color=colors[i])
                        ax2.plot(sec_tr[:, 0], sec_tr[:, 2], color=colors[i])
                        ax1.plot(sec_tr[0, 0], sec_tr[0, 1], 'o',
                                 color=colors[i], markerfacecolor='w')
                        ax2.plot(sec_tr[0, 0], sec_tr[0, 2], 'o',
                                 color=colors[i], markerfacecolor='w')
            else:
                tr.plot_sec(ax1, axes='XY', color=color_sec)
                tr.plot_sec(ax2, axes='XZ', color=color_sec)

    if plot_det_line:
        ax1.plot(det_line[:, 0], det_line[:, 1], '--o', color=color_sec)

    # find UA2 max and min
    UA2_max = np.amax(np.array(A2list))
    UA2_min = np.amin(np.array(A2list))

    ax1.set_title('Ebeam={} keV, UA2:[{}, {}] kV, Btor = {} T, Ipl = {} MA'
                  .format(Ebeam, UA2_min,  UA2_max, Btor, Ipl))
    plt.show()

# %%


def axes_to_indexes(axes):
    symbol_to_index = {'X': 0, 'Y': 1, 'Z': 2}
    return symbol_to_index[axes[0]], symbol_to_index[axes[1]]


def plot_scan_one_subplot(traj_list, geom, Ebeam, Btor, Ipl, ax=None, axes='XY',
                          full_primary=False, plot_analyzer=False,
                          plot_det_line=False, scale=5, color_sec='r',
                          fontsize_axes=14, fontsize_title=16, set_title=True):
    '''
    plot scan for a particular energy Ebeam in chozen plane
    '''
    if ax is None:
        plt.figure()
        ax = plt.gca()
        set_axes_param(ax, '%s, м' %
                       axes[0], '%s, м' % axes[1], fontsize=fontsize_axes)

    # plot geometry
    geom.plot(ax, axes=axes, plot_analyzer=plot_analyzer)

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    A2list = []
    det_line = np.empty([0, 3])

    index1, index2 = axes_to_indexes(axes)

    for tr in traj_list:
        if tr.Ebeam == Ebeam:
            A2list.append(tr.U['A2'])
            if plot_det_line:
                det_line = np.vstack([det_line, tr.RV_sec[0, 0:3]])
            # plot primary
            tr.plot_prim(ax, axes=axes, color='k', full_primary=full_primary)
            # plot secondary
            if plot_analyzer and hasattr(tr, 'RV_sec_slit'):
                n_slits = len(tr.RV_sec_slit)
                for i in range(n_slits):
                    for sec_tr in tr.RV_sec_slit[i]:
                        ax.plot(sec_tr[:, index1],
                                sec_tr[:, index2], color=colors[i])
                        ax.plot(sec_tr[0, index1], sec_tr[0, index2], 'o',
                                color=colors[i], markerfacecolor='w')
            else:
                tr.plot_sec(ax, axes=axes, color=color_sec)

    if plot_det_line:
        ax.plot(det_line[:, index1], det_line[:, index2],
                '--o', color=color_sec)

    # find UA2 max and min
    UA2_max = np.amax(np.array(A2list))
    UA2_min = np.amin(np.array(A2list))
    if set_title:
        ax.set_title('Ebeam={} keV, UA2:[{}, {}] kV, Btor = {} T, Ipl = {} MA'
                     .format(Ebeam, UA2_min,  UA2_max, Btor, Ipl), fontsize=fontsize_title)
    plt.show()

# %%


def plot_fan_one_subplot(traj_list, geom, Ebeam, UA2, Btor, Ipl, ax=None,
                         axes='XY', plot_traj=True, plot_last_points=True,
                         plot_analyzer=False, plot_all=False, full_primary=True,
                         fontsize_axes=14, fontsize_title=16, set_titles=True):

    if ax is None:
        plt.figure()
        ax = plt.gca()
        set_axes_param(ax, '%s, м' %
                       axes[0], '%s, м' % axes[1], fontsize=fontsize_axes)

    index1, index2 = axes_to_indexes(axes)

    # get color cycler
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    colors = cycle(colors)

    # get marker cycler
    markers = cycle(('o', 'v', '^', '<', '>', '*', 'D', 'P', 'd'))

    # plot geometry
    geom.plot(ax, axes='XY', plot_analyzer=plot_analyzer)

    if set_titles:
        ax.set_title('E={} keV, Btor={} T, Ipl={} MA'
                     .format(Ebeam, Btor, Ipl))

    # invert Z ax
    # ax3.invert_xaxis()
    sec_color = 'r'
    marker = 'o'

    for tr in traj_list:
        if plot_all:
            UA2 = tr.U['A2']
            Ebeam_new = tr.Ebeam
            if Ebeam != Ebeam_new:
                Ebeam = Ebeam_new
                sec_color = next(colors)
                marker = next(markers)

        if tr.Ebeam == Ebeam and tr.U['A2'] == UA2:
            if plot_traj:
                # plot primary
                tr.plot_prim(ax, axes=axes, color='k',
                             full_primary=full_primary)

                # plot fan of secondaries
                tr.plot_fan(ax, axes=axes, color=sec_color)

            if plot_last_points:
                last_points = []
                for _traj_points in tr.Fan:
                    last_points.append(_traj_points[-1, :])

                last_points = np.array(last_points)
                if last_points.shape == (0,):
                    print('last_points is empty')
                else:
                    ax.plot(last_points[:, index1], last_points[:, index2],
                            '--', c=sec_color)
                    ax.scatter(last_points[:, index1], last_points[:, index2],
                               marker=marker, c='w', edgecolors=sec_color,
                               label='E={:.1f}, UA2={:.1f}'.format(Ebeam, UA2))

            if (not plot_all) and set_titles:
                ax.set_title('E={} keV, UA2={} kV, UB2={:.1f} kV, '
                             'Btor={} T, Ipl={} MA'
                             .format(Ebeam, UA2, tr.U['B2'], Btor, Ipl))
                break
    ax.legend()
    plt.show()

# %%


def plot_grid(traj_list, geom, Btor, Ipl, onlyE=False,
              linestyle_A2='--', linestyle_E='-',
              marker_A2='*', marker_E='p', **kwargs):
    '''
    plot detector grid in XY and XZ planes
    '''
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

    set_axes_param(ax1, 'X (m)', 'Y (m)')
    set_axes_param(ax2, 'X (m)', 'Z (m)')

    # plot geometry
    geom.plot(ax1, axes='XY')
    geom.plot(ax2, axes='XZ')

    # get the list of A2 and Ebeam
    A2list = []
    Elist = []
    for i in range(len(traj_list)):
        A2list.append(traj_list[i].U['A2'])
        Elist.append(traj_list[i].Ebeam)

    # make sorted arrays of non repeated values
    A2list = np.unique(A2list)
    N_A2 = A2list.shape[0]
    Elist = np.unique(Elist)
    N_E = Elist.shape[0]

    E_grid = np.full((N_A2, 3, N_E), np.nan)
    A2_grid = np.full((N_E, 3, N_A2), np.nan)

    # find UA2 max and min
    UA2_max = np.amax(np.array(A2list))
    UA2_min = np.amin(np.array(A2list))
    # set title
    ax1.set_title('Eb = [{}, {}] keV, UA2 = [{}, {}] kV,'
                  ' Btor = {} T, Ipl = {} MA'
                  .format(traj_list[0].Ebeam, traj_list[-1].Ebeam, UA2_min,
                          UA2_max, Btor, Ipl))

    # make a grid of constant E
    for i_E in range(0, N_E, 1):
        k = -1
        for i_tr in range(len(traj_list)):
            if traj_list[i_tr].Ebeam == Elist[i_E]:
                k += 1
                # take the 1-st point of secondary trajectory
                x = traj_list[i_tr].RV_sec[0, 0]
                y = traj_list[i_tr].RV_sec[0, 1]
                z = traj_list[i_tr].RV_sec[0, 2]
                E_grid[k, :, i_E] = [x, y, z]

        ax1.plot(E_grid[:, 0, i_E], E_grid[:, 1, i_E],
                 linestyle=linestyle_E,
                 marker=marker_E,
                 label=str(int(Elist[i_E]))+' keV', **kwargs)
        ax2.plot(E_grid[:, 0, i_E], E_grid[:, 2, i_E],
                 linestyle=linestyle_E,
                 marker=marker_E,
                 label=str(int(Elist[i_E]))+' keV', **kwargs)
    if onlyE:
        ax1.legend()
        return 0
    # make a grid of constant A2
    for i_A2 in range(0, N_A2, 1):
        k = -1
        for i_tr in range(len(traj_list)):
            if traj_list[i_tr].U['A2'] == A2list[i_A2]:
                k += 1
                # take the 1-st point of secondary trajectory
                x = traj_list[i_tr].RV_sec[0, 0]
                y = traj_list[i_tr].RV_sec[0, 1]
                z = traj_list[i_tr].RV_sec[0, 2]
                A2_grid[k, :, i_A2] = [x, y, z]

        ax1.plot(A2_grid[:, 0, i_A2], A2_grid[:, 1, i_A2],
                 linestyle=linestyle_A2,
                 marker=marker_A2,
                 label=str(round(A2list[i_A2], 1))+' kV')
        ax2.plot(A2_grid[:, 0, i_A2], A2_grid[:, 2, i_A2],
                 linestyle=linestyle_A2,
                 marker=marker_A2,
                 label=str(round(A2list[i_A2], 1))+' kV')

    ax1.legend()
#    ax1.set(xlim=(0.9, 4.28), ylim=(-1, 1.5), autoscale_on=False)
    plt.show()

# %%


def color_by_Ebeam(Ebeam):
    i = Ebeam//20
    return 'C%d' % i


def plot_grid_simple(traj_list, geom, Btor, Ipl, onlyE=False,
                     linestyle_A2='--', linestyle_E='-',
                     marker_A2='*', marker_E='p', ax=None, legend_on=True,
                     title_on=True, color=None, **kwargs):
    '''
    plot detector grid in XY plane
    '''
    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1)

    set_axes_param(ax, 'X (m)', 'Y (m)')

    # plot geometry
    geom.plot(ax, axes='XY')

    # get the list of A2 and Ebeam
    A2list = []
    Elist = []
    for i in range(len(traj_list)):
        A2list.append(traj_list[i].U['A2'])
        Elist.append(traj_list[i].Ebeam)

    # make sorted arrays of non repeated values
    A2list = np.unique(A2list)
    N_A2 = A2list.shape[0]
    Elist = np.unique(Elist)
    N_E = Elist.shape[0]

    E_grid = np.full((N_A2, 3, N_E), np.nan)
    A2_grid = np.full((N_E, 3, N_A2), np.nan)

    # find UA2 max and min
    UA2_max = np.amax(np.array(A2list))
    UA2_min = np.amin(np.array(A2list))
    # set title
    if title_on:
        ax.set_title('Eb = [{}, {}] keV, UA2 = [{}, {}] kV,'
                     ' Btor = {} T, Ipl = {} MA'
                     .format(traj_list[0].Ebeam, traj_list[-1].Ebeam, UA2_min,
                             UA2_max, Btor, Ipl))

    # make a grid of constant E
    for i_E in range(0, N_E, 1):
        k = -1
        for i_tr in range(len(traj_list)):
            if traj_list[i_tr].Ebeam == Elist[i_E]:
                k += 1
                # take the 1-st point of secondary trajectory
                x = traj_list[i_tr].RV_sec[0, 0]
                y = traj_list[i_tr].RV_sec[0, 1]
                z = traj_list[i_tr].RV_sec[0, 2]
                E_grid[k, :, i_E] = [x, y, z]

        if color is None:
            ax.plot(E_grid[:, 0, i_E], E_grid[:, 1, i_E],
                    linestyle=linestyle_E,
                    marker=marker_E,
                    label=str(int(Elist[i_E]))+' keV', color=color_by_Ebeam(Elist[i_E]), **kwargs)
        else:
            ax.plot(E_grid[:, 0, i_E], E_grid[:, 1, i_E],
                    linestyle=linestyle_E,
                    marker=marker_E,
                    label=str(int(Elist[i_E]))+' keV', color=color, **kwargs)
    if onlyE and legend_on:
        ax.legend()
        return 0
    elif onlyE:
        return 0
    # make a grid of constant A2
    for i_A2 in range(0, N_A2, 1):
        k = -1
        for i_tr in range(len(traj_list)):
            if traj_list[i_tr].U['A2'] == A2list[i_A2]:
                k += 1
                # take the 1-st point of secondary trajectory
                x = traj_list[i_tr].RV_sec[0, 0]
                y = traj_list[i_tr].RV_sec[0, 1]
                z = traj_list[i_tr].RV_sec[0, 2]
                A2_grid[k, :, i_A2] = [x, y, z]

        ax.plot(A2_grid[:, 0, i_A2], A2_grid[:, 1, i_A2],
                linestyle=linestyle_A2,
                marker=marker_A2,
                label=str(round(A2list[i_A2], 1))+' kV')
    if legend_on:
        ax.legend()
#    ax1.set(xlim=(0.9, 4.28), ylim=(-1, 1.5), autoscale_on=False)
    plt.show()

# %%


def plot_grid_a3b3(traj_list, geom, Btor, Ipl,
                   marker_E='p'):
    '''
    plot detector grids colored as A3 and B3 voltages
    '''
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharex=True)
    # plot geometry
    geom.plot(ax1, axes='XY', plot_sep=True)
    geom.plot(ax2, axes='XY', plot_sep=True)

    set_axes_param(ax1, 'X (m)', 'Y (m)')
    set_axes_param(ax2, 'X (m)', 'Y (m)')

    # get A2, A3, B3 and E lists
    Elist = np.array([tr.Ebeam for tr in traj_list])
    Elist = np.unique(Elist)
    A2list = np.array([tr.U['A2'] for tr in traj_list])
    A2list = np.unique(A2list)

    A3B3list = np.full((len(traj_list), 2), np.nan)
    k = -1
    for tr in traj_list:
        k += 1
        A3B3list[k, 0] = tr.U['A3']  # choose A3
        A3B3list[k, 1] = tr.U['B3']  # choose B3

    N_E = Elist.shape[0]

    # find UA2 max and min
    UA2_max = np.amax(np.array(A2list))
    UA2_min = np.amin(np.array(A2list))
    # set title
    ax1.set_title('Eb = [{}, {}] keV, UA2 = [{}, {}] kV,'
                  ' Btor = {} T, Ipl = {} MA'
                  .format(traj_list[0].Ebeam, traj_list[-1].Ebeam, UA2_min,
                          UA2_max, Btor, Ipl))

    E_grid = np.full((len(traj_list), 3), np.nan)
    k = -1
    # make a grid of constant E
    for i_E in range(0, N_E, 1):
        for tr in traj_list:
            if abs(tr.Ebeam - Elist[i_E]) < 0.1:
                k += 1
                # take the 1-st point of secondary trajectory
                x = tr.RV_sec[0, 0]
                y = tr.RV_sec[0, 1]
                z = tr.RV_sec[0, 2]
                E_grid[k, :] = [x, y, z]

    # plot grid with A3 coloring
    sc = ax1.scatter(E_grid[:, 0], E_grid[:, 1], s=80,
                     c=A3B3list[:, 0],
                     cmap='jet',
                     marker=marker_E)
    plt.colorbar(sc, ax=ax1, label='A3, kV')

    # plot grid with B3 coloring
    sc = ax2.scatter(E_grid[:, 0], E_grid[:, 1], s=80,
                     c=A3B3list[:, 1],
                     cmap='jet',
                     marker=marker_E)
    plt.colorbar(sc, ax=ax2, label='B3, kV')
    plt.show()


# %%
def plot_traj_toslits(tr, geom, Btor, Ipl, slits=[2],
                      plot_fan=True, plot_flux=True):
    '''
    plot fan of trajectories which go to slits
    '''
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)
    # fig, (ax1, ax3) = plt.subplots(nrows=1, ncols=2)

    set_axes_param(ax1, 'X (m)', 'Y (m)')
    set_axes_param(ax2, 'X (m)', 'Z (m)')
    set_axes_param(ax3, 'Z (m)', 'Y (m)')
    # ax1.set_title('E={} keV, UA2={} kV, Btor={} T, Ipl={} MA'
    #               .format(tr[0].Ebeam, tr[0].U['A2'], Btor, Ipl))

    # plot geometry
    geom.plot(ax1, axes='XY', plot_aim=False, plot_analyzer=True)
    geom.plot(ax2, axes='XZ', plot_aim=False, plot_analyzer=True)
    geom.plot(ax3, axes='ZY', plot_aim=False, plot_analyzer=True)

    n_slits = geom.plates_dict['an'].slits_edges.shape[0]
    # set color cycler
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    colors = colors[:n_slits]
    colors = cycle(colors)

    # plot primary trajectory
    # tr.plot_prim(ax1, axes='XY', color='k', full_primary=True)
    # tr.plot_prim(ax2, axes='XZ', color='k', full_primary=True)
    # tr.plot_prim(ax3, axes='ZY', color='k', full_primary=True)

    # plot precise fan of secondaries
    if plot_fan:
        for fan_tr in tr.fan_to_slits:
            ax1.plot(fan_tr[:, 0], fan_tr[:, 1], color='tab:gray')
            ax2.plot(fan_tr[:, 0], fan_tr[:, 2], color='tab:gray')
            ax3.plot(fan_tr[:, 2], fan_tr[:, 1], color='tab:gray')

    # plot secondaries
    for i_slit in slits:
        c = next(colors)
        for fan_tr in tr.RV_sec_toslits[i_slit]:
            ax1.plot(fan_tr[:, 0], fan_tr[:, 1], color=c)
            ax2.plot(fan_tr[:, 0], fan_tr[:, 2], color=c)
            ax3.plot(fan_tr[:, 2], fan_tr[:, 1], color=c)

    # plot zones
    for i_slit in slits:
        c = next(colors)
        for fan_tr in tr.RV_sec_toslits[i_slit]:
            ax1.plot(fan_tr[0, 0], fan_tr[0, 1], 'o', color=c,
                     markerfacecolor='white')
            ax3.plot(fan_tr[0, 2], fan_tr[0, 1], 'o', color=c,
                     markerfacecolor='white')

    # plot magnetic flux lines
    if plot_flux:
        # !!! import_Bflux function not in hibplib, but in fields.py
        Psi_vals, x_vals, y_vals, bound_flux = fields.import_Bflux('1MA_sn.txt')
        ax1.contour(x_vals, y_vals, Psi_vals, 100)
    plt.show()


# %%
def plot_fatbeam(fatbeam_list, geom, Btor, Ipl, n_slit='all', scale=3):

    if fatbeam_list:
    
        # fig, (ax1, ax3) = plt.subplots(nrows=1, ncols=2)
        fig, axs = plt.subplots(2, 2, sharex='col',  # sharey='row',
                                gridspec_kw={'height_ratios': [scale, 1],
                                             'width_ratios': [scale, 1]})
        ax1, ax2, ax3 = axs[0, 0], axs[1, 0], axs[0, 1]
    
        set_axes_param(ax1, 'X (m)', 'Y (m)')
        set_axes_param(ax2, 'X (m)', 'Z (m)')
        set_axes_param(ax3, 'Z (m)', 'Y (m)')
        
        tr = fatbeam_list
        
        ax1.set_title('E={} keV, UA2={} kV, Btor={} T, Ipl={} MA'
                      .format(tr[0].Ebeam, tr[0].U['A2'], Btor, Ipl))
    
        # get number of slits
        n_slits = geom.plates_dict['an'].n_slits
        # set color cycler
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        colors = colors[:n_slits]
    
        if n_slit == 'all':
            slits = range(n_slits)
        else:
            slits = [n_slit]
    
        # plot trajectories
        for tr in fatbeam_list:
            # plot primary trajectory
            tr.plot_prim(ax1, axes='XY', color='k', full_primary=True)
            tr.plot_prim(ax2, axes='XZ', color='k', full_primary=False)
            tr.plot_prim(ax3, axes='ZY', color='k', full_primary=True)
            # plot first point
            ax1.plot(tr.RV0[0, 0], tr.RV0[0, 1], 'o',
                     color='k', markerfacecolor='white')
            ax2.plot(tr.RV0[0, 0], tr.RV0[0, 2], 'o',
                     color='k', markerfacecolor='white')
            ax3.plot(tr.RV0[0, 2], tr.RV0[0, 1], 'o',
                     color='k', markerfacecolor='white')
    
            # plot secondaries
            for i in slits:
                c = colors[i]
                if tr.RV_sec_toslits[i] is None:
                    continue
                for fan_tr in tr.RV_sec_toslits[i]:
                    ax1.plot(fan_tr[:, 0], fan_tr[:, 1], color=c)
                    ax2.plot(fan_tr[:, 0], fan_tr[:, 2], color=c)
                    ax3.plot(fan_tr[:, 2], fan_tr[:, 1], color=c)
    
                    # plot zones
                    ax1.plot(fan_tr[0, 0], fan_tr[0, 1], 'o', color=c,
                             markerfacecolor='white')
                    ax2.plot(fan_tr[0, 0], fan_tr[0, 2], 'o', color=c,
                             markerfacecolor='white')
                    ax3.plot(fan_tr[0, 2], fan_tr[0, 1], 'o', color=c,
                             markerfacecolor='white')
                    
        # plot geometry: T-15MD wall, plates, analyzer and etc.
        geom.plot(ax1, axes='XY', plot_aim=False, plot_analyzer=True)
        geom.plot(ax2, axes='XZ', plot_aim=False, plot_analyzer=True)
        geom.plot(ax3, axes='ZY', plot_aim=False, plot_analyzer=True)
        
        plt.show()
    else:
        print("plot_fat_beam: traj_list is empty")

# %%
def plot_svs(fatbeam_list, geom, Btor, Ipl, n_slit='all', plot_geom=True,
             plot_prim=True, plot_sec=False, plot_zones=True, plot_cut=False,
             plot_flux=False, plot_legend=False, plot_dots=False,
             alpha_xy=10, alpha_zy=20):
    '''
    plot Sample Volumes
    '''
    
    if fatbeam_list:
        
        fig, (ax1, ax3) = plt.subplots(nrows=1, ncols=2)
        set_axes_param(ax1, 'X (m)', 'Y (m)')
        # set_axes_param(ax2, 'X (m)', 'Z (m)')
        set_axes_param(ax3, 'Z (m)', 'Y (m)')
        tr = fatbeam_list
        ax1.set_title('E={} keV, UA2={} kV, Btor={} T, Ipl={} MA'
                      .format(tr[0].Ebeam, tr[0].U['A2'], Btor, Ipl))
    
        # fig_3d = plt.figure()
        # ax_3d = fig_3d.add_subplot(projection='3d')
        
        # plot geometry
        if plot_geom:
            geom.plot(ax1, axes='XY', plot_aim=False, plot_sep=False, 
                      plot_analyzer=True)
            geom.plot(ax3, axes='ZY', plot_aim=False, plot_analyzer=True)
    
        # get number of slits
        n_slits = geom.plates_dict['an'].n_slits
        # set color cycler
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        colors = colors[:n_slits]
    
        # plot magnetic flux lines
        if plot_flux:
            Psi_vals, x_vals, y_vals, bound_flux = fields.import_Bflux('1MA_sn.txt')
            ax1.contour(x_vals, y_vals, Psi_vals, 100,
                        colors=['k'], linestyles=['-'])
    
        # plot trajectories
        for tr in fatbeam_list:
            # plot primary trajectory
            if plot_prim:
                tr.plot_prim(ax1, axes='XY', color='k', full_primary=True)
                tr.plot_prim(ax3, axes='ZY', color='k', full_primary=True)
            # plot secondaries
            if plot_sec:
                for i in range(n_slits):
                    c = colors[i]
                    for fan_tr in tr.RV_sec_toslits[i]:
                        ax1.plot(fan_tr[:, 0], fan_tr[:, 1], color=c)
                        ax3.plot(fan_tr[:, 2], fan_tr[:, 1], color=c)
    
        if n_slit == 'all':
            slits = reversed(range(n_slits))
        else:
            slits = [n_slit]
            
        # coords of ionization zones for each slit
        coords_list = []
        
        if plot_legend:
            handles = []
        
        # plot sample volumes
        for i in slits:
            c = colors[i]
            coords = np.empty([0, 3])
            coords_first = np.empty([0, 3])
            coords_last = np.empty([0, 3])
            for tr in fatbeam_list:
                # skip empty arrays
                if tr.ion_zones[i] is None:
                    continue
                if tr.ion_zones[i].shape[0] == 0:
                    continue
                coords_first = np.vstack([coords_first, tr.ion_zones[i][0, 0:3]])
                coords_last = np.vstack([coords_last, tr.ion_zones[i][-1, 0:3]])
                # plot zones of each filament
                if plot_dots:
                    ax1.plot(tr.ion_zones[i][:, 0], tr.ion_zones[i][:, 1],
                             'o', color=c, markerfacecolor='white')
                    ax3.plot(tr.ion_zones[i][:, 2], tr.ion_zones[i][:, 1],
                             'o', color=c, markerfacecolor='white')
                    
                    
                    # ax_3d.plot(tr.ion_zones[i][:, 0], tr.ion_zones[i][:, 1], tr.ion_zones[i][:, 2],
                    #           'o', color=c, markerfacecolor='white')
    
            if coords_first.shape[0] == 0:
                coords_list.append(None)
                print(f"Result 3: slit {i+1} has no ionization zone")
                continue
            
            # bad version of polygon filling
            if plot_cut:
                coords = np.vstack([coords_first, coords_last[::-1]])
                coords = np.vstack([coords, coords[0, :]])
                # plot in XY plane
                ax1.fill(coords[:, 0], coords[:, 1], '--', color=c)
                ax1.plot(coords[:, 0], coords[:, 1], color='k', lw=0.5)
                # plot in ZY plane
                ax3.fill(coords[:, 2], coords[:, 1], '--', color=c)
                ax3.plot(coords[:, 2], coords[:, 1], color='k', lw=0.5)
            if plot_zones:
                try:
                    # good version of polygon filling using alphashape
                    coords = np.vstack([coords_first, coords_last])
                    # plot in XY plane
                    hull_xy = alphashape.alphashape(coords[:, [0, 1]], alpha_xy)
                    hull_pts_xy = hull_xy.exterior.coords.xy
                    ax1.fill(hull_pts_xy[0], hull_pts_xy[1], '--', color=c)
                    ax1.plot(hull_pts_xy[0], hull_pts_xy[1], color='k', lw=0.5)
                    
                    # plot in ZY plane
                    hull_zy = alphashape.alphashape(coords[:, [2, 1]], alpha_zy)
                    hull_pts_zy = hull_zy.exterior.coords.xy
                    ax3.fill(hull_pts_zy[0], hull_pts_zy[1], '--', color=c)
                    ax3.plot(hull_pts_zy[0], hull_pts_zy[1], color='k', lw=0.5)
                    
                    # coords_hull = alphashape.alphashape(coords, 0)
                    # # coords_hull = ConvexHull(coords)
                    # hull_pts_xyz = np.asarray(coords_hull.exterior.coords)
                    # hull_pts_xyz = np.asarray(hull_pts_xyz)
                    
                    # x = hull_pts_xyz[:, 0]
                    # y = hull_pts_xyz[:, 1]
                    # z = hull_pts_xyz[:, 2]
                    
                    # # fig = plt.figure()
                    # # ax = fig.add_subplot(111, projection='3d')
                    # ax_3d.plot(x,y,z, 'o', color=c, markerfacecolor='black')
                    # # plt.show()
                                        
                    if len(coords):
                        coords_list.append(coords)
                        
                        if plot_legend:
                            # manually define a new patch 
                            patch = patches.Patch(color=c, label=f'Slit {i+1}')
                            # handles is a list, so append manual patch
                            handles.append(patch)
                            
                    else:
                        coords_list.append(None)
                        print(f"Result 1: slit {i+1} has no ionization zone")
                except:
                    coords_list.append(None)
                    print(f"Result 2: slit {i+1} has no ionization zone")
                    
        if plot_legend:
            ax1.legend(handles=handles, loc='upper right')
            ax3.legend(handles=handles, loc='upper right')
            
        plt.show()
        return coords_list[::-1]
    else:
        print("plot_svs: traj_list is empty")

#%%

def plot_each_sv_3d(coords_list, geomT15, slits_orig):
    """
    
    Plots several figures for each successfully calculated slit.

    Parameters
    ----------
    coords_list : list
        list of ionization zones points for each slit.
    geomT15 : Geometry object
        Geometry of T-15MD.
    slits_orig: int or str
        int: number of slit.
        str: 'all' - all slits.

    Returns
    -------
    log : list
        Contains slit number, size of ionization zone in cartesian,
        plasma coordinates and volume of ionization zone.
        It goes to logfile.

    """
    
    # Pulling cartesian-plasma_coordinates converter function
    T15_R = 1.5 # [m]
    rn = 3
    log_str = None
    volume = 0
    log = []
    
    T15_separatrix = []
    for sep in geomT15.sep:
        T15_separatrix.append(np.array([sep[0]+T15_R, sep[1]]))
    
    coord_converter = CoordConverterDShapeSimple(T15_R, T15_separatrix)
    
    if slits_orig == 'all':
        slits_orig = 0
    
    slit_number = 0 + slits_orig
    for coord in coords_list:
        if coord is not None:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            ax.grid(True)
            
            try:
                hull = ConvexHull(coord)
                volume = ConvexHull(coord).volume * 1e6 # [cm^3] calc volume if iz
                # draw the polygons of the convex hull
                for s in hull.simplices:
                    tri = Poly3DCollection([coord[s]])
                    tri.set_color("blue")
                    tri.set_alpha(0.1)
                    ax.add_collection3d(tri)
                label = f'slit {slit_number+1}\nvolume {round(volume, 2)} cm^3'
                # draw the vertices
                ax.scatter(coord[:, 0], coord[:, 1], coord[:, 2], marker='o', 
                           color='red', label=label)
                
                # ionization zone bounds
                # cartesian
                x_min, x_max = min(coord[:, 0])*100, max(coord[:, 0]*100)
                y_min, y_max = min(coord[:, 1])*100, max(coord[:, 1]*100)
                z_min, z_max = min(coord[:, 2])*100, max(coord[:, 2]*100)
                # plasma coordinates
                coord_plasma = []
                for r in coord:
                    rho, theta, phi = coord_converter(r)
                    coord_plasma.append([rho, theta, phi])
                coord_plasma = np.array(coord_plasma)
                rho_min, rho_max = min(coord_plasma[:, 0]), max(coord_plasma[:, 0])
                theta_min, theta_max = min(coord_plasma[:, 1]), max(coord_plasma[:, 1])
                phi_min, phi_max = min(coord_plasma[:, 2]), max(coord_plasma[:, 2])
                
                log_str = f'slit {slit_number+1}\n' +\
                      'ionization zones sizes\n' +\
                      f'x, y, z, cm: {round(x_max-x_min, rn)}, ' +\
                      f'{round(y_max-y_min, rn)}, ' +\
                      f'{round(z_max-z_min, rn)}\n' +\
                      f'rho, theta, phi: {round(rho_max-rho_min,rn)}, ' +\
                      f'{round(theta_max-theta_min, rn)}, ' +\
                      f'{round(phi_max-phi_min, rn)}\n' +\
                      f'volume: {round(volume, rn)} cm^3\n'
                log.append(log_str)
                print(log_str)
                
                ax.set_xlim(x_min/100, x_max/100)
                ax.set_ylim(y_min/100, y_max/100)
                ax.set_zlim(z_min/100, z_max/100)
                plt.legend()
                plt.show()
            except:
                log.append(f'slit {slit_number+1}\nionization zone is flat (2D)\n')
                print('ionization zone is flat (2D)')
        slit_number +=1
    return log


# %%
def plot_legend(ax, figure_name):
    '''
    plots legend in separate window
    ax : axes to get legnd from
    figure_name : get figure's name as base for legend's file name
    return : figure object
    '''
    # create separate figure for legend
    figlegend = plt.figure(num='Legend_for_' + figure_name, figsize=(1, 12))
    # get legend from ax
    figlegend.legend(*ax.get_legend_handles_labels(), loc="center")
    plt.show()
    return figlegend


# %%
def plot_sec_angles(traj_list, Btor, Ipl, Ebeam='all', linestyle='-o'):
    '''
    plot grid colored as angles at the last point of the secondary trajectory
    '''

    # plotting params
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    set_axes_param(ax1, 'UA2 (kV)', r'Exit $\alpha$ (grad)')
    set_axes_param(ax2, 'UA2 (kV)', r'Exit $\beta$ (grad)')

    if Ebeam == 'all':
        equal_E_list = np.array([tr.Ebeam for tr in traj_list])
        equal_E_list = np.unique(equal_E_list)
    else:
        equal_E_list = np.array([float(Ebeam)])

    angles_dict = {}

    for Eb in equal_E_list:
        angle_list = []
        for tr in traj_list:
            if tr.Ebeam == Eb:
                # Get vector coords for last point in Secondary traj
                Vx = tr.RV_sec[-1, 3]  # Vx
                Vy = tr.RV_sec[-1, 4]  # Vy
                Vz = tr.RV_sec[-1, 5]  # Vz

                angle_list.append(
                    [tr.U['A2'], tr.U['B2'],
                     np.arctan(Vy/np.sqrt(Vx**2 + Vz**2))*180/np.pi,
                     np.arctan(-Vz/Vx)*180/np.pi])

        angles_dict[Eb] = np.array(angle_list)
        ax1.plot(angles_dict[Eb][:, 0], angles_dict[Eb][:, 2],
                 linestyle, label=str(Eb))
        ax2.plot(angles_dict[Eb][:, 0], angles_dict[Eb][:, 3],
                 linestyle, label=str(Eb))

    ax1.legend()
    ax2.legend()
    ax1.axis('tight')
    ax2.axis('tight')
    plt.show()
    return angles_dict

# %%


def plot_lam(traj_list, rho_interp, Ebeam='all', slits=range(7)):
    '''
    plot SV size along trajectory (lambda) vs UA2 and vs rho
    '''
    # plotting params
    fig1, ax1 = plt.subplots()
    fig1, ax2 = plt.subplots()
    set_axes_param(ax1, 'UA2 (kV)', r'$\lambda$ (mm)')
    set_axes_param(ax2, r'$\rho', r'$\lambda$ (mm)')

    if Ebeam == 'all':
        equal_E_list = np.array([tr.Ebeam for tr in traj_list])
        equal_E_list = np.unique(equal_E_list)
    else:
        equal_E_list = np.array([float(Ebeam)])

    for Eb in equal_E_list:
        for i_slit in slits:
            UA2_list = []
            rho_list = []
            lam_list = []
            for tr in traj_list:
                if tr.Ebeam == Eb and tr.ion_zones[i_slit].shape[0] > 0:
                    UA2_list.append(tr.U['A2'])
                    rho_list.append(rho_interp(tr.RV_sec[0, :3])[0])
                    lam_list.append(np.linalg.norm(tr.ion_zones[i_slit][0]
                                                   - tr.ion_zones[i_slit][-1])*1000)
            ax1.plot(UA2_list, lam_list, '-o', label='slit ' + str(i_slit+1))
            ax2.plot(rho_list, lam_list, '-o', label='slit ' + str(i_slit+1))
    ax1.legend()
    ax2.legend()
    plt.show()


# %%
def plot_fan3d(traj_list, geom, Ebeam, UA2, Btor, Ipl,
               azim=0.0, elev=0.0,
               plot_slits=True, plot_all=True):

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.grid(True)

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    colors = cycle(colors)
    # plot aim dot
    # r_aim = geom.r_dict['aim']
    # ax.plot(r_aim[0], r_aim[1], r_aim[2], '*')

    # # plot slit dot
    # r_aim = geom.r_dict['slit']
    # ax.plot(r_aim[0], r_aim[1], r_aim[2], '*')

    for tr in traj_list:
        if plot_all:
            UA2 = tr.U['A2']
            sec_color = next(colors)
        else:
            sec_color = 'r'

        if tr.Ebeam == Ebeam and tr.U['A2'] == UA2:
            # plot primary
            ax.plot(tr.RV_prim[:, 0], tr.RV_prim[:, 1],
                    tr.RV_prim[:, 2], color='k')

            last_points = []
            for i in tr.Fan:
                ax.plot(i[:, 0], i[:, 1], i[:, 2], color=sec_color)
                last_points.append(i[-1, :])
            last_points = np.array(last_points)
            ax.plot(last_points[:, 0], last_points[:, 1],
                    last_points[:, 2], '--o', color=sec_color)

            ax.set_title('E={} keV, UA2={} kV, UB2={:.1f} kV, '
                         'Btor={} T, Ipl={} MA'
                         .format(Ebeam, UA2, tr.U['B2'], Btor, Ipl))
            # plt.show()
            if not plot_all:
                break

    for name in geom.plates_dict.keys():
        if name == 'A2':
            color = 'b'
        elif name == 'B2':
            color = 'r'
        else:
            color = 'tab:gray'
        for i in range(2):
            x = list(geom.plates_dict[name].edges[i][:, 0])
            y = list(geom.plates_dict[name].edges[i][:, 1])
            z = list(geom.plates_dict[name].edges[i][:, 2])
            verts = [list(zip(x, y, z))]
            poly = Poly3DCollection(verts, edgecolors='k',
                                    facecolors=color)
            poly.set_alpha(0.5)
            ax.add_collection3d(poly)

    # plot slits
    if plot_slits:
        r_slits = geom.slits_edges
        n_slits = r_slits.shape[0]
        for i in range(n_slits):
            x = list(r_slits[i, 1:, 0])
            y = list(r_slits[i, 1:, 1])
            z = list(r_slits[i, 1:, 2])
            verts = [list(zip(x, y, z))]
            poly = Poly3DCollection(verts, edgecolors='k',
                                    facecolors='b')
            poly.set_alpha(0.5)
            ax.add_collection3d(poly)

    # set position
    ax.view_init(elev=elev, azim=azim)

# %%


def plot_RV_prim(traj_list, geom, Ebeam, UA2, Btor, Ipl, plot_analyzer=False):

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)
    # fig, (ax1, ax3) = plt.subplots(nrows=1, ncols=2)

    set_axes_param(ax1, 'X (m)', 'Y (m)')
    set_axes_param(ax2, 'X (m)', 'Z (m)')
    set_axes_param(ax3, 'Z (m)', 'Y (m)')

    # get color cycler
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    colors = cycle(colors)
    # get marker cycler
    markers = cycle(('o', 'v', '^', '<', '>', '*', 'D', 'P', 'd'))

    # plot geometry
    geom.plot(ax1, axes='XY', plot_analyzer=plot_analyzer)
    geom.plot(ax2, axes='XZ', plot_analyzer=plot_analyzer)
    geom.plot(ax3, axes='ZY', plot_analyzer=plot_analyzer)

    ax1.set_title('E={} keV, Btor={} T, Ipl={} MA'
                  .format(Ebeam, Btor, Ipl))

    sec_color = 'r'
    marker = 'o'

    # plot RV_prim
    for tr in traj_list:
        if tr.Ebeam == Ebeam and tr.U['A2'] == UA2:
            ax1.plot(tr.RV_prim[:, 0], tr.RV_prim[:, 1],
                     '--', c=sec_color)
            ax2.plot(tr.RV_prim[:, 0], tr.RV_prim[:, 2],
                     '--', c=sec_color)
            ax3.plot(tr.RV_prim[:, 2], tr.RV_prim[:, 1],
                     '--', c=sec_color)

    # ax1.legend()
    plt.show()
