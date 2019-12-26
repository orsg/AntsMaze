import PhaseSpace, Mazes, StateCalculator, StateMachine, MazeTrajectory
import glob
import matplotlib.pyplot as plt

mazes = {'XLT': {"name":Mazes.MAZE_T_XL,
                 "bias": (-0.1,0.25,-90),
                 "factor": (1,1,-1)},
        'SLT': {"name":Mazes.MAZE_T_SL,
                 "bias": (0,0,90),
                 "factor": (1,1,1)},
                }
CURRENT_MAZE = 'XLT'
# CURRENT_MAZE = 'SLT'


ps = PhaseSpace.PhaseSpace(mazes[CURRENT_MAZE]["name"], 0.2, 3, (12, 21), (0, 15), name=CURRENT_MAZE)
# p = PhaseSpace(MAZE_SPECIAL, 0.25, 6, (12,37), (0,20), name='Special')
# p = PhaseSpace(MAZE_LONG, 0.25, 3, (5,22), (0,20), name='Long')
ps.load_space(ps.name + ".pkl")
# ps.calculate_boundary()
# ps.save_space(ps.name+".pkl")

sc = StateCalculator.StateCalculator(ps)
sc.load(ps.name + "_states.pkl")
sc.calculate_states(recalculate_volume=True)
sc.save(ps.name + "_states.pkl")

sm = StateMachine.StateMachine(sc.state_ids, sc.state_dict, ps)
sm.calculate_state_machine()
# sm.visualize()

# ps.visualize_space()
sc.plot_state_map()
sc.plot_interactive_states()

tr = {}
traj_paths = glob.glob("Trajectories\{}*.mat".format(CURRENT_MAZE))
for i, t in enumerate(traj_paths):
    tr[t] = MazeTrajectory.MazeTrajectory(sc.state_ids, ps, sm, t,
                                          coords_bias=mazes[CURRENT_MAZE]["bias"],
                                          coords_factor=mazes[CURRENT_MAZE]["factor"])
    # ps.plot_trajectory(tr[t].traj, color=(0, 0, 1.0 / (i+1)))

tr[t].animate(delay=50, initial_frame=800, skip_rate=15)



input("tap to finish")


