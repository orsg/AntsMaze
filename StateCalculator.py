import itertools
import pickle
import numpy as np
from shapely import geometry
from matplotlib import pyplot as plt
from matplotlib.widgets import TextBox, Slider
from tqdm import tqdm
from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage import binary_dilation
import cc3d

VOLUME_TAG = 'V'
EMPTY_STATE_LEN = len('bpo') * 4

def _deserialize_state(s):
    is_volume = False
    if s.startswith(VOLUME_TAG):
        s = s[len(VOLUME_TAG):]
        is_volume=True
    elif s == "Illegal":
        return {'board':{'points':set(), 'lines':set()}, 'load': {'points':set(), 'lines':set()}}
    bp = s.find('bpo')
    bl = s.find('bli')
    lp = s.find('lpo')
    ll = s.find('lli')
    return {'board': {
                'points': set([int(x) for x in s[bp+3:bl].split("-") if len(x) > 0]),
                'lines': set([tuple(int(y) for y in x.split(',')) for x in s[bl+3:lp].split("-") if len(x) > 0])},
            'load': {
                'points': set([int(x) for x in s[lp+3:ll].split("-") if len(x) > 0]),
                'lines': set([tuple(int(y) for y in x.split(',')) for x in s[ll+3:].split("-") if len(x) > 0])},
            'is_volume': is_volume,
    }


def serialize_state(state_dict, board_dual_points):
    # used to gather close points into one point
    for dual_points in board_dual_points:
        if dual_points in state_dict["board"]["lines"]:
            state_dict["board"]["lines"].remove(dual_points)
            state_dict["board"]["points"].update(dual_points)
        if dual_points[0] in state_dict["board"]["points"] or dual_points[1] in state_dict["board"]["points"]:
            state_dict["board"]["points"].update(dual_points)
    return "{}{}{}{}{}{}{}{}".format("bpo", "-".join(["{:02d}".format(i) for i in state_dict['board']['points']]),
                                     "bli", "-".join(["{:02d},{:02d}".format(i, j) for (i, j) in state_dict['board']['lines']]),
                                     'lpo', "-".join(["{:02d}".format(i) for i in state_dict['load']['points']]),
                                     "lli", "-".join(["{:02d},{:02d}".format(i, j) for (i, j) in state_dict['load']['lines']]))


class State(object):
    states_list = []

    def __init__(self, name, rep_point, phase_space):
        self.name = name
        self.index = -1
        self.ps = phase_space
        self.is_volume = False
        if self.name.startswith(VOLUME_TAG):
            self.is_volume=True
        for s in self.__class__.states_list:
            if s['states'][0].name == name:
                s['counter'] += 1
                self.index = s['counter']
                s['states'].append(self)
                break
        if self.index == -1:
            self.index = 0
            self.__class__.states_list.append({'counter': self.index,
                                               'states': [self]})
        self.rep_point = rep_point

    @property
    def edges(self):
        return _deserialize_state(self.name)

    def display_markers(self, ax):
        shapes = self.edges
        for c in shapes['load']['points']:
            coords = self.ps.maze.load.shape.coords[c]
            ax.plot(coords[0], coords[1], 'go', markersize=8)
        for c1, c2 in shapes['load']['lines']:
            coords = np.array(self.ps.maze.load.shape.coords[c1:c1 + 2]).T
            ax.plot(coords[0], coords[1], 'r', markersize=6)
        for c in shapes['board']['points']:
            coords = self.ps.maze.board.shape.coords[c]
            ax.plot(coords[0], coords[1], 'go', markersize=8)
        for c1, c2 in shapes['board']['lines']:
            coords = np.array(self.ps.maze.board.shape.coords[c1:c1 + 2]).T
            ax.plot(coords[0], coords[1], 'r', markersize=6)

    def visualize(self, ax=None, point=None):
        if ax is None:
            ax = plt.subplot()
        if point is None:
            point = self.ps.indexes_to_coords(*(self.rep_point))
        self.ps.maze.load.translate(point[0], point[1])
        self.ps.maze.load.rotate(point[2])
        self.ps.maze.visualize(ax=ax)
        if type(self.name) is str and self.name.startswith('b'):
            self.display_markers(ax)
        ax.set_xlim((self.ps.shape['x'][0] - 5, self.ps.shape['x'][1] + 5))
        ax.set_ylim((self.ps.shape['y'][0] - 5, self.ps.shape['y'][1] + 5))
        plt.draw()

class StateCalculator(object):
    # Factor to multiply resolution in order to calculate the max distance
    # that still counts as touching. Here taking diagonal factor
    ERROR_DISTANCE_FACTOR = 1  # 1.2 * pow(2, 0.5)

    def __init__(self, phase_space, max_volume_dist=1.5, board_dual_points=[]):
        self.ps = phase_space
        self.touch_err_distance = self.ps.pos_resolution * self.ERROR_DISTANCE_FACTOR
        self._boundary_states_names = None
        self._total_states_names = None
        self.states = None
        self.max_volume_dist = max_volume_dist
        self.state_ids = None
        self.state_dict = None
        self._board_dual_points = board_dual_points

    def calculate_states(self, recalculate_volume=False):
        # Compute all boundary points in the phase Space
        if self.ps.space_boundary is None:
            self.ps.calculate_boundary()
        if self._boundary_states_names is None:
            self._boundary_states_names = np.zeros(self.ps.space_boundary.shape, object)
            self._name_boundary_points()
        if self.states is None or recalculate_volume:
            self.states = np.zeros(self.ps.space_boundary.shape, type(State))
            self._total_states_names = np.zeros(self.ps.space_boundary.shape, object)
            self._append_volume_to_states()
            self._group_to_states_by_connectivity()

    def name_point_by_touch_type(self, ix, iy, itheta):
        load = self.ps.maze.load
        shape_dict = {'load': load, 'board': self.ps.maze.board}
        x, y, theta = self.ps.indexes_to_coords(ix, iy, itheta)
        load.translate(x, y)
        load.rotate(theta)
        corners = {'board': {'lines': set(), 'points': set()}, 'load': {'lines': set(), 'points': set()}}
        for s1_name, s2_name in itertools.permutations(['board', 'load']):
            s1, s2 = shape_dict[s1_name], shape_dict[s2_name]
            for i2, line in enumerate(s2.iterate_lines()):
                l1, l2 = line.coords[0], line.coords[1]
                C = np.array(l1) - np.array(l2)
                for i1, point in enumerate(s1.shape.coords):
                    point = geometry.Point(point)
                    d_line = point.distance(line)
                    if d_line < self.touch_err_distance:
                        if 0<i1<len(s1.shape.coords)-1:
                            p0, p1, p2 = [np.array(s1.shape.coords[j]) for j in [i1-1, i1, i1+1]]
                            A = p0 - p1
                            B = p2 - p1
                            if ((float(np.cross(A, C)) * float(np.cross(A, B)) > 0 and\
                                    float(np.cross(B, C)) * float(np.cross(B, A)) > 0) or\
                                (float(np.cross(A, -C)) * float(np.cross(A, B)) > 0 and\
                                 float(np.cross(B, -C)) * float(np.cross(B, A)) > 0)) and (theta % 90 != 0):
                                #TODO: the theta%90 will work only for mazes with 90 degress angles only
                                # the line cannot realy 'touch' the point, ignore it
                                continue
                        if point.distance(geometry.Point(l1)) == d_line or \
                                point.distance(geometry.Point(l2)) == d_line:
                                    continue
                            # another edge case to handle
                        if s2_name == 'board' and (i2,i2+1) in self._board_dual_points:
                            continue
                        corners[s2_name]['lines'].add((i2, i2 + 1))
                        corners[s1_name]['points'].add(i1)
        state_name = serialize_state(corners, self._board_dual_points)
        if len(state_name) <= EMPTY_STATE_LEN:
            # print "Error: no corners for: {},{},{}".format(x, y, theta)
            pass
        else:
            self._boundary_states_names[ix, iy, itheta] = state_name

    def _name_boundary_points(self):
        # Name all corners of each load and board
        print("StatesCalculator: naming boundary states")
        with tqdm(total=np.count_nonzero(self.ps.space_boundary)) as progress_bar:
            for ix, iy, itheta in self.ps.iterate_space_index():
                if self.ps.space_boundary[ix, iy, itheta]:
                    self.name_point_by_touch_type(ix, iy, itheta)
                    progress_bar.update(1)

    def save(self, path='ps_states.pkl'):
        pickle.dump((self._boundary_states_names, self._total_states_names, self.states, self.state_ids, self.state_dict), open(path, 'wb'))

    def load(self, path='ps_states.pkl'):
        (self._boundary_states_names, self._total_states_names, self.states, self.state_ids, self.state_dict) = pickle.load(open(path, 'rb'))

    def _fix_to_permissive_connectivity(self, labeld_state_map, cc_outoput, permissivness=4):
        new_id = 0
        new_labeld_state_map = np.zeros(labeld_state_map.shape)
        # for each id that has degeneracy
        print("StateCalculator: Aplying permissive connectivity algorithm")
        for id in tqdm(np.unique(labeld_state_map)):
            # create 3d binary map
            deg_map = np.where(labeld_state_map == id, 1, 0)
            sample_index = tuple(np.argwhere(deg_map == 1)[0])
            if np.count_nonzero(deg_map) == np.count_nonzero(cc_outoput == cc_outoput[sample_index]):
                # non degenerate state
                new_labeld_state_map = np.where(deg_map, new_id, new_labeld_state_map)
                new_id += 1
                continue
            # degenerate case
            dilated_map = binary_dilation(deg_map, iterations=permissivness)
            # run cc again
            new_cc = cc3d.connected_components(dilated_map, connectivity=26)
            for id in np.unique(new_cc):
                new_labeld_state_map = np.where((new_cc==id) & deg_map, new_id, new_labeld_state_map)
                new_id += 1
        return new_labeld_state_map


    def _group_to_states_by_connectivity(self):
        # Prepare to cc3d format
        state_names = np.unique(self._total_states_names[self._total_states_names.nonzero()])
        state_to_label_dict = {v: i for i, v in enumerate(state_names)}
        state_to_label_dict[0] = 10000
        labeld_state_map = np.zeros(self._total_states_names.shape, dtype=np.int32)
        for ix, iy, itheta in self.ps.iterate_space_index():
            labeld_state_map[ix, iy, itheta] = state_to_label_dict[self._total_states_names[ix, iy, itheta]]
        self.state_ids = cc3d.connected_components(labeld_state_map, connectivity=26)
        label_to_state_dict = {i: v for i, v in enumerate(state_names)}
        label_to_state_dict[10000] = "Illegal"

        self.state_ids = self._fix_to_permissive_connectivity(labeld_state_map, self.state_ids)

        # Create the new states
        self.state_dict = {}
        for id in np.unique(self.state_ids):
            coords = np.where(self.state_ids == id)
            point = (coords[0][0], coords[1][0], coords[2][0])
            for temp_point in zip(coords[0], coords[1], coords[2]):
                # look for points which are on the boundary for state representation
                if self.ps.space_boundary[temp_point]:
                    point = temp_point
                    break
            state_label = labeld_state_map[point]
            state_name = label_to_state_dict[state_label]
            self.state_dict[id] = State(state_name, point, self.ps)

        # update the state array
        for ix, iy, itheta in self.ps.iterate_space_index():
            if self._total_states_names[ix, iy, itheta]:
                self.states[ix, iy, itheta] = self.state_dict[self.state_ids[ix, iy, itheta]]

    def _append_volume_to_states(self):
        neutral_state_name = "neutral"
        print("StatesCalculator: calculating volume states")
        for itheta in tqdm(range(self._boundary_states_names.shape[2])):
            # init with the maximum distance possible
            dist_map = np.ones(self._boundary_states_names.shape[0:2]) * max(self._boundary_states_names.shape[0:2]) * 100
            state_map = np.zeros(dist_map.shape, type(State))
            for state in np.unique(self._boundary_states_names[:, :, itheta][self._boundary_states_names[:, :, itheta].nonzero()]):
                # calculate distance matrix for specific theta. boundaries are not taken care
                # since for every volume point the closest boundary distant will be in smaller
                # than any distance which crosses the allowed space boundary
                # set 0 for dist source in cc3d
                state_dist_map = np.where(self._boundary_states_names[:, :, itheta] == state, 0, 1)
                state_dist_map = distance_transform_edt(state_dist_map)
                state_map = np.where(np.logical_and(dist_map > state_dist_map,
                                                    state_dist_map > 0),
                                     state, state_map)
                dist_map = np.where(np.logical_and(dist_map > state_dist_map,
                                                   state_dist_map > 0),
                                    state_dist_map, dist_map)

            # update states array
            self._total_states_names[:, :, itheta] = np.where(self.ps.space[:, :, itheta] == 1,
                                                              state_map, self._total_states_names[:, :, itheta])
            self._total_states_names[:, :, itheta] = np.where(self._boundary_states_names[:, :, itheta] != 0,
                                                              self._boundary_states_names[:,:,itheta], self._total_states_names[:, :, itheta])
            # neutral state
            self._total_states_names[:, :, itheta] = np.where(np.logical_and(np.logical_and(self._boundary_states_names[:, :, itheta] == 0,
                                                                       self._total_states_names[:, :, itheta] != 0), dist_map > self.max_volume_dist),
                                                               VOLUME_TAG + self._total_states_names[:, :, itheta].astype(str).astype(object), self._total_states_names[:, :, itheta])


    def visualize_point(self, ipoint, ax=None):
        if ax is None:
            ax = plt.subplot()

        point = self.ps.indexes_to_coords(*ipoint)
        self.ps.maze.load.translate(point[0], point[1])
        self.ps.maze.load.rotate(point[2])
        self.ps.maze.visualize(ax=ax)
        state = self.states[ipoint]
        if state and type(state.name) is str and len(state.name) > 1:
            state.display_markers(ax)
        palette = plt.get_cmap("gist_ncar")
        palette.set_bad(alpha=0.0)
        img = np.swapaxes(self.state_ids[:, :, ipoint[2]], 0, 1)
        ax.imshow(np.ma.masked_where(img == 1, img), extent=(self.ps.shape['x'][0],
                                                             self.ps.shape['x'][1],
                                                             self.ps.shape['y'][0],
                                                             self.ps.shape['y'][1]), origin='lower',
                  cmap=palette, vmin=0, vmax=np.max(img))
        ax.set_xlim(self.ps.shape['x'])
        ax.set_ylim(self.ps.shape['y'])
        ax.set_xlim((self.ps.shape['x'][0] - 5, self.ps.shape['x'][1] + 5))
        ax.set_ylim((self.ps.shape['y'][0] - 5, self.ps.shape['y'][1] + 5))
        plt.draw()
        plt.pause(0.0001)

    def plot_interactive_states(self, state_list=None):
        self.cur_s = 0
        self.cur_pos = 0
        if state_list is None:
            state_list = State.states_list
        self.states_to_plot = [s['states'] for s in state_list if not s['states'][0].name.startswith(VOLUME_TAG)]

        def plot_state():
            self.states_ax.cla()
            self.states_fig.suptitle(
                "State Number: %d. Current pos:%d Number of positions:%d. Total state names: %d, Total_states:%d" % (self.cur_s,
                                                                                               self.cur_pos,
                                                                                               state_list[self.cur_s]['counter']+1,
                                                                                               len(state_list),
                                                                                               sum([state_list[i]['counter']+1 for i in range(len(state_list))]))
            )
            self.states_to_plot[self.cur_s][self.cur_pos].visualize(self.states_ax)
            self.states_fig.canvas.draw()
            # plt.draw()

        def key_event(e):
            if e.key == "right":
                self.cur_s = self.cur_s + 1
            elif e.key == "left":
                self.cur_s = self.cur_s - 1
            elif e.key == "up":
                self.cur_pos = self.cur_pos + 1
            elif e.key == "down":
                self.cur_pos = self.cur_pos - 1
            else:
                return
            self.cur_s = self.cur_s % len(self.states_to_plot)
            self.cur_pos = self.cur_pos % len(self.states_to_plot[self.cur_s])
            self.state_slider.set_val(self.cur_s)

        def state_submit(val):
            i = int(val)
            if self.cur_s != i:
                self.cur_pos = 0
            self.cur_s = i % len(self.states_to_plot)
            plot_state()

        self.states_fig = plt.figure(figsize=(40, 40))
        # state slider
        slide_color = 'lightgoldenrodyellow'
        ax_slide = plt.axes([0.25, 0.05, 0.45, 0.03], facecolor=slide_color)
        self.state_slider = Slider(ax_slide, 'state_id', 0, len(state_list) - 1,
                                   valinit=0, valstep=1)
        self.state_slider.on_changed(state_submit)

        # key press
        self.states_fig.canvas.mpl_connect('key_press_event', key_event)
        self.states_ax = self.states_fig.add_subplot(111)
        plt.axis('scaled')
        plot_state()

    def plot_state_map(self, state_list=None):
        self.load_itheta, self.load_ix, self.load_iy = 0, 0, 0

        def plot_state():
            self.theta_ax.cla()
            ipoint = (self.load_ix, self.load_iy, self.load_itheta)
            self.theta_fig.suptitle("State id:{}   X:{}, Y:{}, Theta: {}.".format(self.state_ids[ipoint],
                                                                                  *(self.ps.indexes_to_coords(*ipoint))))

            self.visualize_point(ipoint, self.theta_ax)
            self.theta_fig.canvas.draw()
            # plt.draw()

        def key_event(e):
            if e.key == "right":
                self.load_itheta = self.load_itheta + 1
            elif e.key == "left":
                self.load_itheta = self.load_itheta - 1
            else:
                return
            self.load_itheta = self.load_itheta % self.states.shape[2]
            self.theta_slider.set_val(self.load_itheta)

        def theta_submit(val):
            self.load_itheta = int(val) % self.states.shape[2]
            plot_state()

        def onclick(event):
            x, y = event.xdata, event.ydata
            if x is None or y is None:
                return
            if not (self.ps.shape['x'][0] <= x <= self.ps.shape['x'][1] and \
                    self.ps.shape['y'][0] <= y <= self.ps.shape['y'][1]):
                return
            self.load_ix = int((x - self.ps.shape['x'][0]) / self.ps.pos_resolution)
            self.load_iy = int((y - self.ps.shape['y'][0]) / self.ps.pos_resolution)
            plot_state()

        self.theta_fig = plt.figure(figsize=(40, 40))

        # theta slider
        ax_theta = plt.axes([0.25, 0.05, 0.45, 0.03])
        self.theta_slider = Slider(ax_theta, 'theta', 0, self.states.shape[2] - 1,
                                   valinit=0, valstep=1)
        self.theta_slider.on_changed(theta_submit)

        # mouse click
        self.theta_fig.canvas.mpl_connect('button_press_event', onclick)
        # key press
        self.theta_fig.canvas.mpl_connect('key_press_event', key_event)
        self.theta_ax = self.theta_fig.add_subplot(111)
        plt.axis('scaled')
        plot_state()


## TODO: rare cases should have smaller "volume potential"


