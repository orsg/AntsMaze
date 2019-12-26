from StateCalculator import *
from scipy.io import loadmat
import matplotlib.animation as animation
import matplotlib.pyplot as plt

class MazeTrajectory(object):

    OUT_OF_SCOPE_POINT = (0,0,0)

    def __init__(self, state_id_matrix, phase_space, state_machine, traj_path, coords_bias=(0,0,90)):
        self.sim = state_id_matrix
        self.ps = phase_space
        self.sm = state_machine
        self.coords_bias = coords_bias
        self._load_traj(traj_path)
        self.traj_indices = np.apply_along_axis(self._calculate_state_ids, 0, self.traj)
        # fix traj indices that are out of boundaries
        self.traj_indices = np.apply_along_axis(self._fix_point, 0, self.traj_indices)
        self.traj_state_ids = self.sim[tuple(self.traj_indices.tolist())]

    def _calculate_state_ids(self, c):
        ipos = self.ps.coords_to_indexes(c[0], c[1], c[2])
        if not 0<=ipos[0]<self.sim.shape[0] or not 0<=ipos[1]<self.sim.shape[1] or not 0<=ipos[2]<self.sim.shape[2]:
            return self.OUT_OF_SCOPE_POINT
        return ipos

    def _fix_point(self, p):
        if p[0] == self.OUT_OF_SCOPE_POINT[0] and p[1] == self.OUT_OF_SCOPE_POINT[1] and p[2] == self.OUT_OF_SCOPE_POINT[2]:
            # point is out of scope
            return p
        if self.ps.space[tuple(p)]:
            # Allowed point
            return p
        # The point is out of boundaries. find closest point in x-y and assign it:
        boundary_points = np.array(np.nonzero(self.ps.space[:, :, p[2]])).T
        return np.append(boundary_points[np.argmin(np.linalg.norm(boundary_points - p[:2], axis=1))], p[2])

    def _load_traj(self, traj_path):
        matfile = loadmat(traj_path)
        load_center = matfile['load_center']
        load_orientation = matfile['shape_orientation']
        self.traj = np.concatenate([load_center, load_orientation], axis=1).T
        self.traj = (self.traj.T + self.coords_bias).T
        self.traj[2] = self.traj[2] % 360


    def update_frame(self, i):
        i = (i * self.anim_skip_rate + self.anim_initial_frame) % (self.traj.shape[1])
        self.anim_ax.cla()
        state = self.sm.state_dict[self.traj_state_ids[i]]
        state.visualize(self.anim_ax, point=self.traj[:, i])
        self.anim_ax.set_title("state id:{}, frame:{}".format(self.traj_state_ids[i], i))
        self.anim_ax.set_xlim((0, self.ps.shape['x'][1] + 10))
        self.anim_ax.set_ylim((0, self.ps.shape['y'][1] + 10))

    def animate(self, delay=20, initial_frame=0, skip_rate=1):
        self.anim_initial_frame = initial_frame
        self.anim_skip_rate = skip_rate
        self.anim_fig = plt.figure()
        self.anim_ax = self.anim_fig.add_subplot(111)

        ani = animation.FuncAnimation(
            self.anim_fig, self.update_frame, frames=range(self.traj.shape[1]), interval=delay, cache_frame_data=False)
        plt.draw()
        plt.pause(0.001)




    #process traj - smooth it.
    #normalize the traj with the traj length
    #generate state list for the traj
    #calculate transfer statistics for the traj - probabilities,
    #