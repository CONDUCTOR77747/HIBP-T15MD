# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 20:20:57 2024

@author: conductor
"""

# if calculate_zones:
    
    
#     # new_path = 'filaments_test'
#     # filaments = np.arange(3, 21, 3)
#     # n_gammas = np.arange(3, 21, 3)
#     # print(f'Filaments: {filaments[0]}-{filaments[-1]}')
#     # print(f'Filaments: {n_gammas[0]}-{n_gammas[-1]}')
#     # print(f'Total operations: {len(filaments)*len(n_gammas)}')
#     # for fil in filaments:
#     #     for ng in n_gammas:
            
#         # using optimized trajectories
#         traj_list_optimized = copy.deepcopy(traj_list_passed)
    
#         # fatbeam_calc function parameters
#         fatbeam_args = [traj_list_optimized,
#                         E,
#                         B,
#                         geomT15,
#                         Btor,
#                         Ipl]
        
#         mode = 'save' # 'load' and 'save' modes
#         if mode == 'save':
#             load_traj = False
#             save_traj = True
#             plot_trajs = True
#             rescale_plots = False
#             close_plots = True
#             save_plots = True
#         elif mode == 'load':
#             load_traj = True
#             save_traj = False
#             plot_trajs = True
#             rescale_plots = False
#             close_plots = False
#             save_plots = False
        
#         fatbeam_kwargs = {'Ebeam_orig':'260',
#                           'UA2_orig':'10',
#                           'target':'slit',
#                           'slits_orig':'4',
#                           'd_beam':0.02,
#                           'foc_len':50,
#                           'n_filaments_xy':5,
#                           'n_gamma':5,
#                           'timestep_divider':20,
#                           'dt':2e-8,
#                           'calc_mode':'cpu_unparallel', # cpu_unparallel
#                           'load_traj':load_traj,
#                           'save_traj':save_traj,
#                           'path_orig': os.path.join('fatbeam', 'results', '2024'),
#                           'plot_trajs': plot_trajs,
#                           'rescale_plots': rescale_plots,
#                           'close_plots': close_plots,
#                           'save_plots': save_plots,
#                           'create_table': False}
        
#         with Timer() as time:
#             traj_fat_beam = hb.fatbeam_calc(*fatbeam_args, **fatbeam_kwargs)

#         print(f'Fatbeam executed: {round(time.elapsed, 2)} sec')
        
#%%

# def plot_fan_on_slits(geomT15):
#     geomT15.slits

# using optimized trajectories
# traj_list_optimized = copy.deepcopy(traj_list_passed)
# hbplot.plot_fan(traj_list_optimized, geomT15, 240, 14, Btor, Ipl, plot_analyzer=True)

#%%
traj_list_optimized = copy.deepcopy(traj_list_passed)
fatbeam = fb.Fatbeam(traj_list_optimized[0], E, B, geomT15, Btor, Ipl)
fatbeam._set_new_RV0s(0.02, 50, n=7)
fatbeam.plot3d()
r_aim = geomT15.r_dict['r0'] # geomT15.r_dict['an']

fatbeam_passed = fatbeam.pass_to_slits(traj_list_optimized[0], dt, E, B, geomT15, target='slit', timestep_divider=1,
                  slits=[3], no_intersect=True, no_out_of_bounds=True,
                  print_log=True)

hbplot.plot_fatbeam(fatbeam_passed, geomT15, Btor, Ipl, n_slit='3', scale=3)

# fatbeam.filaments[0].pass_fan(r_aim, E, B, geomT15, no_intersect=True, no_out_of_bounds=True, fan_step_divider=1)

# hbplot.plot_fan([fatbeam.filaments[0]], geomT15, 100, 6, Btor, Ipl, plot_traj=True,
#               plot_last_points=True, plot_analyzer=True, plot_all=False,
#               full_primary=False)

# fatbeam.traj.pass_fan(r_aim, E, B, geomT15, no_intersect=True, no_out_of_bounds=True)

# hbplot.plot_fan([fatbeam.traj], geomT15, 100, 6, Btor, Ipl, plot_traj=True,
#               plot_last_points=True, plot_analyzer=True, plot_all=False,
#               full_primary=False)

# hbplot.plot_traj([fatbeam.fatbeam[0]], geomT15, 100, 6, Btor, Ipl, full_primary=False,
#               plot_analyzer=True, subplots_vertical=False, scale=5)
#%%
# slits_edges = geomT15.plates_dict['an'].slits_edges

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.set_xlabel('X (m)')
# ax.set_ylabel('Y (m)')
# ax.set_zlabel('Z (m)')
# ax.grid(True)



# for slits in slits_edges:
#     for edge in slits:
#         print(edge)
#         # x, y, z = edge[0], edge[1], edge[2]
#         # ax.scatter(x,y,z)

# slits_spot = geomT15.plates_dict['an'].slits_spot        
# slits_plane_n = geomT15.plates_dict['an'].slit_plane_n
# r_an = geomT15.r_dict['slit']

# for point in slits_spot:
#     print(point)
#     x, y, z = point[0], point[1], point[2]
#     ax.scatter(x,y,z)
#     xn, yn, zn = slits_plane_n[0], slits_plane_n[1], slits_plane_n[2]
#     xr, yr, zr = r_an[0], r_an[1], r_an[2]
#     ax.quiver(xr, yr, zr, xn, yn, zn, length=0.05)