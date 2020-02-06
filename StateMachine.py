from StateCalculator import *
import matplotlib.pyplot as plt
import networkx as nx

class StateMachine(object):

    def __init__(self, state_id_matrix, state_dict, phase_space):
        """
        :param state_matrix: 3d matrix with state id (int) in each cell.
        :param state_obj_matrix: 3d matrix with state id (int) in each cell.
        """
        self.sim = state_id_matrix
        self.ps = phase_space
        self.state_dict = state_dict
        self.AREA_GAUGE_FACTOR = {'xy': 1,
                                  'xt': self.ps.theta_cell_normalization,
                                  'yt': self.ps.theta_cell_normalization}
        self.connection_dict = None
        self.graph = None
        self.end_nodes = None

    def _build_graph(self):
        self.graph = nx.DiGraph()
        volumes = np.unique(self.sim, return_counts=True)
        [self.graph.add_node(i) for i in self.connection_dict.keys()]
        nx.set_node_attributes(self.graph, {i: {"volume": count, "is_start": False, "is_end": False, 'shortest_dist': None} for i, count in zip(list(volumes[0]), list(volumes[1]))})
        for i, cons in self.connection_dict.items():
            sum_weights = sum(cons.values())
            for j, weight in cons.items():
                # normalize each node's weight to one
                self.graph.add_edges_from([(i, j)], weight=weight, weight_normed=weight/sum_weights, traj_counter=0, traj_normed=0)

    def calculate_state_machine(self):
        self.connection_dict = {i : {} for i in np.unique(self.sim)}
        print("StateMachine: building graph")
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
        volumes = list(dict(self.graph.nodes.data('volume')).values())
        scalars = [(v / max(volumes))*4 + 5 for v in volumes]

        mlab.figure()
        mlab.clf()

        pts = mlab.points3d(xyz[:, 0], xyz[:, 1], xyz[:, 2],
                            # scale_factor=0.03,
                            # scale_mode='none',
                            colormap='winter',
                            resolution=20)

        pts.mlab_source.dataset.lines = np.array(list(self.graph.edges()))
        tube = mlab.pipeline.tube(pts, tube_radius=0.01)
        mlab.pipeline.surface(tube, color=(0.8, 0.8, 0.8))

    def _update_traj(self, state_ids_list):
        for s1, s2 in zip(state_ids_list[:-1], state_ids_list[1:]):
            if s1 != s2:
                try:
                    self.graph.edges[s1,s2]['traj_counter'] += 1
                except:
                    # print("Failed with:{},{}".format(s1,s2))
                    continue

    def load_trajectories(self, maze_trajs):
        for maze_traj in maze_trajs:
            self._update_traj(maze_traj.traj_state_ids)
            self._update_traj(maze_traj.dual_traj_state_ids)
        # update each node sum to 1
        for node in self.graph.nodes:
            edges = list(self.graph.edges.data('traj_counter', nbunch=node))
            sum_edges = sum([x[2] for x in edges])
            if sum_edges == 0:
                # print("No data for node: %d" % (node))
                continue
            for edge in edges:
                self.graph.edges[edge[0], edge[1]]['traj_normed'] = edge[2] / sum_edges

    #
    # def _calculate_end_state_map(self):
    #     skfmm.distance()

    def set_end_states(self, end_states_map):
        illegal = 1000000
        self.end_nodes = set()
        for state_id in np.unique(np.where(end_states_map, self.sim, illegal)):
            if state_id == illegal:
                continue
            self.graph.nodes[state_id]['is_end'] = True
            self.end_nodes.add(state_id)
        self._compute_shortest_paths()

    def _compute_shortest_paths(self):
        self._shortest_paths = dict(nx.all_pairs_shortest_path_length(self.graph))
        for node in self.graph.nodes:
            if self.graph.nodes[node]['is_end']:
                self.graph.nodes[node]['shortest_dist'] = 0
                continue
            distances = self._shortest_paths[node]
            self.graph.nodes[node]['shortest_dist'] = min([distances[k] for k in distances.keys() if self.graph.nodes[k]['is_end']])

