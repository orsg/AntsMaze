import PhaseSpace, Mazes, StateCalculator, StateMachine, MazeTrajectory
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import pickle
import seaborn as sns

mazes = {'XLT': {"name":Mazes.MAZE_T_XL,
                 "bias": (0,0,90),
                 "factor": (1,1,1),
                 'dual_points': [(1,2), (7,8)]},
        'SLT': {"name":Mazes.MAZE_T_SL,
                 "bias": (0,0,90),
                 "factor": (1,1,1),
                 'dual_points': [(1,2), (7,8)]},
         'LT': {"name": Mazes.MAZE_T_L,
                 "bias": (0, 0, 90),
                 "factor": (1, 1, 1),
                 'dual_points': [(1, 2), (7, 8)]},
         'MT': {"name": Mazes.MAZE_T_M,
                 "bias": (0, 0, 90),
                 "factor": (1, 1, 1),
                 'dual_points': [(1, 2), (7, 8)]},
                }

def maze_T_dual_trajectory_transform(point):
    return (point[0], Mazes.BOARD_SLIT_HIEGHT - point[1], ((-1)*(point[2] - 90) + 90)  % 360)

def get_T_end_map(ps):
    for ix, iy, itheta in ps.iterate_space_index():
        x, y, theta = ps.indexes_to_coords(ix,iy,itheta)
        if ps.space[ix,iy,itheta] and x > Mazes.BOARD_SLIT_WIDTH+2:
            ps.maze.load.translate(x,y)
            ps.maze.load.rotate(theta)
            min_x = min([c[0] for c in ps.maze.load.shape.coords])
            if min_x > Mazes.BOARD_SLIT_WIDTH:
                map[ix,iy,itheta]=1
    return map

def analyze_maze(maze_name, data_dir='saved_data', data_exist=False, plot=False):
    ps = PhaseSpace.PhaseSpace(mazes[maze_name]["name"], 0.2, 3, (12, 23), (2, 13), name=maze_name)
    # p = PhaseSpace(MAZE_SPECIAL, 0.25, 6, (12,37), (0,20), name='Special')
    # p = PhaseSpace(MAZE_LONG, 0.25, 3, (5,22), (0,20), name='Long')
    if data_exist:
        ps.load_space(os.path.join(data_dir, ps.name+".pkl"))
    else:
        ps.calculate_boundary()
    ps.save_space(os.path.join(data_dir, ps.name+".pkl"))

    sc = StateCalculator.StateCalculator(ps, board_dual_points=mazes[maze_name]["dual_points"])
    if data_exist:
        sc.load(os.path.join(data_dir, ps.name + "_states.pkl"))
    sc.calculate_states(recalculate_volume=True)
    sc.save(os.path.join(data_dir, ps.name + "_states.pkl"))

    tr = {}
    distances = []
    traj_paths = glob.glob("trajectories\{}*.pkl".format(maze_name))
    for i, t in enumerate(traj_paths):
        try:
            tr[t] = MazeTrajectory.MazeTrajectory(sc.state_ids, ps, sc.state_dict, t,
                                                  coords_bias=mazes[maze_name]["bias"],
                                                  coords_factor=mazes[maze_name]["factor"],
                                                  dual_traj_transform=maze_T_dual_trajectory_transform)
            distances.append({"name":t, 'cm':tr[t].cm_dist, 'rot':tr[t].rot_dist, 'states_dist': tr[t].states_dist})
        except:
            continue
        print(i)
        # ps.plot_trajectory(tr[t].traj, color=(0, 0, 1.0 / (i+1)))

    df = pd.DataFrame(distances)
    df['board']=maze_name
    df['scale']=ps.maze.load.centroid_max_dist/Mazes.MAZE_T_XL.load.centroid_max_dist
    df['rotational_dist(scaled)']=df['rot']/df['scale']
    df['centroid_dist(scaled)'] = df['cm'] / df['scale']
    df['centroid+rotational_dist(scaled)']=df['rotational_dist(scaled)']+df['centroid_dist(scaled)']
    pickle.dump(df, open(os.path.join(data_dir, maze_name + "results.pkl"), 'wb'))
    # sm.visualize()
    if plot:
        ps.visualize_space()
        sc.plot_state_map()
        sc.plot_interactive_states()
    return (df, ps, sc, tr)
    # tr[traj_paths[i]].animate(delay=50, initial_frame=0, skip_rate=15)

    # sm = StateMachine.StateMachine(sc.state_ids, sc.state_dict, ps)
    # sm.calculate_state_machine()
    # sm.load_trajectories(tr.values())
    # sm.set_end_states(ends_map)

dfs=[]
results={}
for m in mazes:
    print("Analayzing: {}".format(m))
    df, ps, sc, tr = analyze_maze(m, data_exist=True)
    results[m]={'ps':ps, 'sc':sc, 'tr':tr}
    dfs.append(df)
df = pd.concat(dfs)
df = df[df.cm!=np.inf]
print(df.groupby('board').count())
print(df.groupby('board').mean())
print(df.groupby('board').var())

measures = ['centroid_dist(scaled)', 'rotational_dist(scaled)', 'centroid+rotational_dist(scaled)']
df_melted = df[['board']+measures].melt(id_vars='board').rename(columns=str.title)
plt.figure()
sns.barplot("Variable", 'Value', hue="Board", data=df_melted)
plt.figure()
sns.barplot('board', "states_dist", data=df)
plt.pause(0.1)

for measure in measures:
    plt.figure()
    sns.barplot('board', measure, data=df)




n = 15
for m in mazes:
    results[m]['ps'].maze.load.translate(5,5)
    results[m]['ps'].maze.load.rotate(0)
    results[m]['ps'].maze.visualize()
    [l.plot_trajectory() for l in list(results[m]['tr'].values())[:n]]
input("tap to finish")


special = PhaseSpace.PhaseSpace(Mazes.MAZE_SPECIAL, 0.25, 3, (12,37), (0,20), name='Special')
special.load_space("Special.pkl")
ixl = PhaseSpace.PhaseSpace(Mazes.MAZE_I_XL, 0.25, 3, (12,23), (2,13), name='IXL')
ixl.load_space("iXL_space.pkl")
hxl = PhaseSpace.PhaseSpace(Mazes.MAZE_H_XL, 0.25, 3, (12,23), (2,13), name='HXL')
hxl.load_space("HXL_space.pkl")
