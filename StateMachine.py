from StateCalculator import *

class StateMachine(object):

    def __init__(self, state_id_matrix, state_obj_matrix, phase_space):
        """
        :param state_matrix: 3d matrix with state id (int) in each cell.
        :param state_obj_matrix: 3d matrix with state id (int) in each cell.
        """
        self.sim = state_id_matrix
        self.som = state_obj_matrix
        self.ps = phase_space
        self.graph = None

    def calculate_state_machine(self):
        if self.graph is not None:
            return
        for ix, iy, itheta in self.ps.iterate_space():
            for n_ix
            state_id = self.sim[ix,iy,itheta]



    #process traj - smooth it.
    #normalize the traj with the traj length
    #generate state list for the traj
    #calculate transfer statistics for the traj - probabilities,
    #