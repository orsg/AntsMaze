import PhaseSpace, Mazes, StateCalculator, StateMachine

p = PhaseSpace.PhaseSpace(Mazes.MAZE_T_SL, 0.2, 6, (12, 21), (0, 15), name='ps')
# p = PhaseSpace(MAZE_SPECIAL, 0.25, 6, (12,37), (0,20), name='Special')
# p = PhaseSpace(MAZE_LONG, 0.25, 3, (5,22), (0,20), name='Long')
p.load_space(p.name + ".pkl")
# p.calculate_boundary()
# p.save_space(p.name+".pkl")

sc = StateCalculator.StateCalculator(p)
sc.load(p.name + "_states.pkl")
sc.calculate_states()
# sc.save(p.name + "_states.pkl")

sm = StateMachine.StateMachine(sc.cc, sc.states, sc.state_dict, p)
sm.calculate_state_machine()
sm.visualize()

p.visualize_space()
sc.plot_state_map()
sc.plot_interactive_states()


