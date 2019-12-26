import PhaseSpace, Mazes, StateCalculator, StateMachine, MazeTrajectory
import glob
import matplotlib.pyplot as plt

CURRENT_MAZE = 'SLT'

ps = PhaseSpace.PhaseSpace(Mazes.MAZE_T_SL, 0.2, 6, (12, 21), (0, 15), name='ps')
# p = PhaseSpace(MAZE_SPECIAL, 0.25, 6, (12,37), (0,20), name='Special')
# p = PhaseSpace(MAZE_LONG, 0.25, 3, (5,22), (0,20), name='Long')
ps.load_space(ps.name + ".pkl")
# p.calculate_boundary()
# p.save_space(p.name+".pkl")

sc = StateCalculator.StateCalculator(ps)
sc.load(ps.name + "_states.pkl")
sc.calculate_states()
# sc.save(p.name + "_states.pkl")

sm = StateMachine.StateMachine(sc.cc, sc.state_dict, ps)
sm.calculate_state_machine()
sm.visualize()

ps.visualize_space()
# sc.plot_state_map()
# sc.plot_interactive_states()

tr = {}
traj_paths = glob.glob("Trajectories\{}*.mat".format(CURRENT_MAZE))
for i, t in enumerate(traj_paths):
    tr[t] = MazeTrajectory.MazeTrajectory(sc.cc, ps, sm, t,
                                          coords_bias=(0,0,90))
    ps.plot_trajectory(tr[t].traj, color=(0, 0, 1.0 / (i+1)))

tr[t].animate(delay=50, initial_frame=800, skip_rate=15)



input("tap to finish")


