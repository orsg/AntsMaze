from StateCalculator import *
from scipy.io import loadmat
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from scipy import signal
from collections import Counter
import Mazes

def occurence_filter_1d(a, window_size):
    b = np.copy(a)
    for i in range(window_size//2, a.shape[0]-window_size//2):
        b[i] = Counter(a[i - window_size//2:i + window_size//2+1]).most_common(1)[0][0]
    return b


class MazeTrajectory(object):

    OUT_OF_SCOPE_POINT = (0,0,0)
    #TODO: change for different boards than slit
    FPS = 50

    def __init__(self, state_id_matrix, phase_space, state_id_dict, traj_path, coords_bias=(0,0,90), coords_factor=(1,1,1), dual_traj_transform=None):
        self.sim = state_id_matrix
        self.ps = phase_space
        self.state_dict = state_id_dict
        self._load_traj(traj_path, coords_bias, coords_factor)
        self.traj_indices = np.apply_along_axis(self._calculate_state_ids, 0, self.traj)
        # fix traj indices that are out of boundaries
        self.traj_indices = np.apply_along_axis(self._fix_point, 0, self.traj_indices)
        self.traj_state_ids = self.sim[tuple(self.traj_indices.tolist())]
        self.traj_state_ids = occurence_filter_1d(self.traj_state_ids, self.FPS*2+1)

        self._create_dual_traj(dual_traj_transform)
        self.dist_ratio = self.ps.maze.load.centroid_max_dist / Mazes.MAZE_T_XL.load.centroid_max_dist
        self._init_virtual_maze_contstants()
        self._calculate_distance()


    def _init_virtual_maze_contstants(self):
        # define virtual edges to handle scaling issue. based on XL size
        self.VIRTUAL_X_LEFT_EDGE = 8 * self.dist_ratio
        self.VIRTUAL_X_END_FROM_SLIT = 1.2 * self.dist_ratio
        self.VIRTUAL_Y_DIST_FROM_BOARD = (Mazes.BOARD_SLIT_HIEGHT / 2 - Mazes.MAZE_T_XL.load.centroid_max_dist) * self.dist_ratio
        self.VIRTUAL_X_START_POINT_FROM_SLIT = Mazes.MAZE_T_XL.load.centroid_max_dist * self.dist_ratio

    def _create_dual_traj(self, dual_traj_transform=None):
        if dual_traj_transform is None:
            self.dual_traj, self._dual_traj_indices, self.dual_traj_state_ids = None, None, None
            return
        else:
            self.dual_traj = np.apply_along_axis(dual_traj_transform, 0, self.traj)
            self.dual_traj_indices = np.apply_along_axis(self._calculate_state_ids, 0, self.dual_traj)
            self.dual_traj_indices = np.apply_along_axis(self._fix_point, 0, self.dual_traj_indices)
            self.dual_traj_state_ids = self.sim[tuple(self.dual_traj_indices.tolist())]

    def _get_states_dist(self, s1_id, s2_id):
        s1, s2 = self.state_dict[s1_id], self.state_dict[s2_id]
        # ignore 90 degrees cases
        #TODO: revisit for mazes with border with different angles
        if self.ps.indexes_to_coords(*tuple(s1.rep_point))[2] % 90 == 0 or \
                self.ps.indexes_to_coords(*tuple(s2.rep_point))[2] % 90 == 0:
            return 0
        if s1.edges['is_volume'] and not s2.edges['is_volume']:
            return 1
        if s1.edges['board'] == s2.edges['board'] or s2.edges['is_volume']:
            return 0
        if s2.edges['board']['lines'].issubset(s1.edges['board']['lines']) and \
                s2.edges['board']['points'].issubset(s1.edges['board']['points']):
            return 0
        return 1

    def _calculate_distance(self):
        self.cm_dist = 0
        self.rot_dist = 0
        self.states_dist = 0
        finished = False
        started = False
        for i in range(self.traj.shape[1] - 1):
            p1, p2 = self.traj[:,i], self.traj[:,i + 1]
            s1, s2 = self.traj_state_ids[i], self.traj_state_ids[i+1]
            if not p2[0] > self.VIRTUAL_X_LEFT_EDGE or\
                    tuple(self.traj_indices[:,i]) == self.OUT_OF_SCOPE_POINT or tuple(self.traj_indices[:,i+1]) == self.OUT_OF_SCOPE_POINT or\
                    not (Mazes.BOARD_SLIT_HIEGHT/2 - self.VIRTUAL_Y_DIST_FROM_BOARD < p2[1] < Mazes.BOARD_SLIT_HIEGHT/2 + self.VIRTUAL_Y_DIST_FROM_BOARD):
                # irrelevant point - out of counting scope
                continue
            if (not started) and p2[0] < (Mazes.BOARD_SLIT_WIDTH - self.VIRTUAL_X_START_POINT_FROM_SLIT):
                #haven't started yet, ignore this point
                continue
            started=True
            if p2[0] > Mazes.BOARD_SLIT_WIDTH + self.VIRTUAL_X_END_FROM_SLIT*self.dist_ratio:
                # object reached the end
                finished=True
                break
            self.cm_dist += np.linalg.norm(p2[0:2]-p1[0:2])
            dtheta=min((p2[2]-p1[2])%360, (360-(p2[2]-p1[2]))%360)
            if dtheta > 10:
                print("Error with dtheta: {}".format(dtheta))
            self.rot_dist += dtheta * self.ps.theta_distance_factor
            self.states_dist += self._get_states_dist(s1,s2)

        if not finished:
            print("Didn't finish maze!!!")
            self.cm_dist = np.inf
            self.rot_dist = np.inf
            self.states_dist = np.inf



    def _calculate_state_ids(self, c):
        ipos = self.ps.coords_to_indexes(c[0], c[1], c[2])
        if not 0<=ipos[0]<self.sim.shape[0] or not 0<=ipos[1]<self.sim.shape[1] or not 0<=ipos[2]<self.sim.shape[2]:
            return self.OUT_OF_SCOPE_POINT
        return ipos

    def _fix_point(self, p):
        if tuple(p) == self.OUT_OF_SCOPE_POINT:
            # point is out of scope
            return p
        if self.ps.space[tuple(p)]:
            # Allowed point
            return p
        # The point is out of boundaries. find closest point in x-y and assign it:
        boundary_points = np.array(np.nonzero(self.ps.space[:, :, p[2]])).T
        return np.append(boundary_points[np.argmin(np.linalg.norm(boundary_points - p[:2], axis=1))], p[2])

    def _load_traj(self, traj_path, bias, factor):
        if traj_path.endswith('mat'):
            matfile = loadmat(traj_path)
            load_center = matfile['load_center']
            load_orientation = matfile['shape_orientation']
        else:
            load_center, load_orientation = pickle.load(open(traj_path, 'rb'))
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

    def plot_trajectory(self):
        ax = plt.subplot()
        self.ps.maze.board.visualize(ax)
        ax.plot(self.traj[0], self.traj[1])

    def animate(self, delay=50, initial_frame=0, skip_rate=10, is_dual=False):
        self.anim_initial_frame = initial_frame
        self.anim_skip_rate = skip_rate
        self.anim_fig = plt.figure()
        self.anim_ax = self.anim_fig.add_subplot(111)
        self.anim_dual = is_dual

        ani = animation.FuncAnimation(
            self.anim_fig, self.update_frame, frames=range(self.traj.shape[1]), interval=delay, cache_frame_data=False)
        plt.draw()
        plt.pause(0.001)

    def interactive_traj_map(self):
        self.cur_frame=0

        def plot_frame():
            self.interactive_ax.cla()
            self.interactive_fig.suptitle("Frame_id:{} State id:{}   X:{:.2f}, Y:{:.2f}, Theta: {:.2f}.".format(self.cur_frame, self.traj_state_ids[self.cur_frame],
                                                                                                    *(tuple(self.traj[:,self.cur_frame]))))
            self.state_dict[self.traj_state_ids[self.cur_frame]].visualize(self.interactive_ax,
                                                                           point=tuple(self.traj[:,self.cur_frame]))
            self.interactive_ax.set_xlim((0, self.ps.shape['x'][1] + 10))
            self.interactive_ax.set_ylim((0, self.ps.shape['y'][1] + 10))
            self.interactive_fig.canvas.draw()

        def key_event(e):
            if e.key == "right":
                self.cur_frame += 1 % self.traj_state_ids.shape[0]
            elif e.key == "left":
                self.cur_frame -= 1 % self.traj_state_ids.shape[0]
            elif e.key == "up":
                self.cur_frame += 20 % self.traj_state_ids.shape[0]
            elif e.key == "down":
                self.cur_frame -= 20 % self.traj_state_ids.shape[0]
            else:
                return
            self.frame_slider.set_val(self.cur_frame)

        def frame_submit(val):
            self.cur_frame = int(val) % self.traj_state_ids.shape[0]
            plot_frame()

        self.interactive_fig = plt.figure(figsize=(40, 40))

        # theta slider
        ax_slider = plt.axes([0.25, 0.05, 0.45, 0.03])
        self.frame_slider = Slider(ax_slider, 'frame_id', 0, self.traj_state_ids.shape[0] - 1,
                                   valinit=0, valstep=1)
        self.frame_slider.on_changed(frame_submit)
        # key press
        self.interactive_fig.canvas.mpl_connect('key_press_event', key_event)
        self.interactive_ax = self.interactive_fig.add_subplot(111)
        plt.axis('scaled')
        plot_frame()
