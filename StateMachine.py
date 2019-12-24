from StateCalculator import *
import matplotlib.pyplot as plt
import networkx as nx

class StateMachine(object):

    def __init__(self, state_id_matrix, state_obj_matrix, state_dict, phase_space):
        """
        :param state_matrix: 3d matrix with state id (int) in each cell.
        :param state_obj_matrix: 3d matrix with state id (int) in each cell.
        """
        self.sim = state_id_matrix
        self.som = state_obj_matrix
        self.ps = phase_space
        self.state_dict = state_dict
        self.AREA_GAUGE_FACTOR = {'xy': 1,
                                  'xt': self.ps.theta_cell_normalization,
                                  'yt': self.ps.theta_cell_normalization}
        self.connection_dict = None
        self.graph = None

    def _build_graph(self):
        self.graph = nx.Graph()
        [self.graph.add_node(i) for i in self.connection_dict.keys()]
        for i, cons in self.connection_dict.items():
            for j, weight in cons.items():
                self.graph.add_weighted_edges_from([(i, j, weight)])

    def calculate_state_machine(self):
        self.connection_dict = {i : {} for i in np.unique(self.sim)}
        [self._update_machine_with_cell(ix, iy, itheta) for ix, iy, itheta in self.ps.iterate_space_index()]
        self._build_graph()

    def _states_step(self, ix, iy, itheta, n_ix, n_iy, n_itheta, face_type):
        state_id = self.sim[ix, iy, itheta]
        n_state_id = self.sim[n_ix, n_iy, n_itheta]
        if state_id == n_state_id:
            return
        # states are different
        if not n_state_id in self.connection_dict[state_id].keys():
            self.connection_dict[state_id][n_state_id] = 0
        self.connection_dict[state_id][n_state_id] += 1 * self.AREA_GAUGE_FACTOR[face_type]

    def _update_machine_with_cell(self, ix, iy, itheta):
        # iterate only over adjacent neighbours
        for delta in [-1, 1]:
            # y - theta face
            if 0 <= ix + delta < self.sim.shape[0]:
                self._states_step(ix, iy, itheta, ix+delta, iy, itheta, "yt")
            # x - theta face
            if 0 <= iy + delta < self.sim.shape[1]:
                self._states_step(ix, iy, itheta, ix, iy+delta, itheta, "xt")
            # x - y face
            if 0 <= itheta + delta < self.sim.shape[2]:
                self._states_step(ix, iy, itheta, ix, iy, itheta+delta, "xy")

    def visualize(self, ax=None):
        if ax is None:
            ax = plt.subplot()
        pos = nx.shell_layout(self.graph)
        # nx.draw(self.graph, pos=pos, with_labels=True, font_weight='bold')
        labels = nx.get_edge_attributes(self.graph, 'weight')
        # nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=labels)

        from mayavi import mlab

        # 3d spring layout
        # numpy array of x,y,z positions in sorted node order
        xyz = np.array([self.state_dict[v].rep_point for v in sorted(self.graph)])
        # scalar colors
        scalars = np.array(list(self.graph.nodes())) + 5

        mlab.figure(1, bgcolor=(0, 0, 0))
        mlab.clf()

        pts = mlab.points3d(xyz[:, 0], xyz[:, 1], xyz[:, 2],
                            scalars,
                            scale_factor=0.1,
                            scale_mode='none',
                            color = (0.2,0.2,0.2),
                            resolution=20)

        pts.mlab_source.dataset.lines = np.array(list(self.graph.edges()))
        tube = mlab.pipeline.tube(pts, tube_radius=0.01)
        mlab.pipeline.surface(tube, color=(0.8, 0.8, 0.8))
        bla


    #process traj - smooth it.
    #normalize the traj with the traj length
    #generate state list for the traj
    #calculate transfer statistics for the traj - probabilities,