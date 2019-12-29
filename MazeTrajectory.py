from StateCalculator import *
from scipy.io import loadmat
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from scipy import signal
class MazeTrajectory(object):

    OUT_OF_SCOPE_POINT = (0,0,0)

    def __init__(self, state_id_matrix, phase_space, state_id_dict, traj_path, coords_bias=(0,0,90), coords_factor=(1,1,1), dual_traj_transform=None):
        self.sim = state_id_matrix
        self.ps = phase_space
        self.state_dict = state_id_dict
        self._load_traj(traj_path, coords_bias, coords_factor)
        self.traj_indices = np.apply_along_axis(self._calculate_state_ids, 0, self.traj)
        # fix traj indices that are out of boundaries
        self.traj_indices = np.apply_along_axis(self._fix_point, 0, self.traj_indices)
        self.traj_state_ids = self.sim[tuple(self.traj_indices.tolist())]
        self._create_dual_traj(dual_traj_transform)

    def _create_dual_traj(self, dual_traj_transform=None):
        if dual_traj_transform is None:
            self.dual_traj, self._dual_traj_indices, self.dual_traj_state_ids = None, None, None
            return
        else:
            self.dual_traj = np.apply_along_axis(dual_traj_transform, 0, self.traj)
            self.dual_traj_indices = np.apply_along_axis(self._calculate_state_ids, 0, self.dual_traj)
            self.dual_traj_indices = np.apply_along_axis(self._fix_point, 0, self.dual_traj_indices)
            self.dual_traj_state_ids = self.sim[tuple(self.dual_traj_indices.tolist())]

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

    def _load_traj(self, traj_path, bias, factor):
        matfile = loadmat(traj_path)
        load_center = matfile['load_center']
        load_orientation = matfile['shape_orientation']
        self.traj = np.concatenate([load_center, load_orientation], axis=1).T
        self.traj = (self.traj.T * factor + bias).T
        self.traj[2] = self.traj[2] % 360
        # Smooth the orientation
        for i in range(1, self.traj[2].shape[0]):
            if 250 > abs((self.traj[2,i] - self.traj[2,i-1]) % 360) > 90:
                self.traj[2, i] = (self.traj[2, i] - 180) % 360
        self.traj[2] = signal.medfilt(self.traj[2], kernel_size=3)


    def update_frame(self, i):
        traj, state_ids = (self.traj, self.traj_state_ids) if not self.anim_dual else (self.dual_traj, self.dual_traj_state_ids)
        i = (i * self.anim_skip_rate + self.anim_initial_frame) % (self.traj.shape[1])
        self.anim_ax.cla()
        state = self.state_dict[state_ids[i]]
        state.visualize(self.anim_ax, point=traj[:, i])
        self.anim_ax.set_title("state id:{}, frame:{}".format(state_ids[i], i))
        self.anim_ax.set_xlim((0, self.ps.shape['x'][1] + 10))
        self.anim_ax.set_ylim((0, self.ps.shape['y'][1] + 10))

    def animate(self, delay=20, initial_frame=0, skip_rate=1, is_dual=False):
        self.anim_initial_frame = initial_frame
        self.anim_skip_rate = skip_rate
        self.anim_fig = plt.figure()
        self.anim_ax = self.anim_fig.add_subplot(111)
        self.anim_dual = is_dual

        ani = animation.FuncAnimation(
            self.anim_fig, self.update_frame, frames=range(self.traj.shape[1]), interval=delay, cache_frame_data=False)
        plt.draw()
        plt.pause(0.001)




    #process traj - smooth it.
    #normalize the traj with the traj length
    #generate state list for the traj
    #calculate transfer statistics for the traj - probabilities,
    #