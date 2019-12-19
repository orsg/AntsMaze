from StateCalculator import *

class MazeTrjectory(object):

    def __init__(self, state_c, trajectory):
        self.sc = state_c
        self.traj = trajectory


    #process traj - smooth it.
    #normalize the traj with the traj length
    #generate state list for the traj
    #calculate transfer statistics for the traj - probabilities,
    #