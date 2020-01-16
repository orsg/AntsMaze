import numpy as np
from Mazes import *
# from mpl_toolkits.mplot3d import Axes3D
import itertools
import pickle
from scipy.io import loadmat
from mayavi import mlab
from tqdm import tqdm


class PhaseSpace(object):
    MAX_THETA = 360

    def __init__(self, maze, pos_resolution, theta_resolution,
                 x_range, y_range, theta_range=(0, MAX_THETA),
                 name=""):
        """
        :param board_coords:
        :param load_coords:
        :param pos_resolution: load replacement resolution (in in the coords units)
        :param theta_resolution: theta replace
        :param x_range: tuple of the x-space range, in the coords units
        :param y_range: tuple of the y-space range, in the coords units
        """
        self.maze = maze
        self.name = name
        self.pos_resolution = pos_resolution
        self.theta_resolution = theta_resolution
        self.shape = {'x': x_range,
                      'y': y_range,
                      'theta': theta_range}
        self.x_size = x_range[1] - x_range[0]
        self.y_size = y_range[1] - y_range[0]
        self.theta_size = theta_range[1] - theta_range[0]
        self.theta_factor = float(self.x_size) / self.theta_size
        self.space = None
        self.space_boundary = None
        # self._initialize_maze_edges()

    @property
    def theta_cell_normalization(self):
        return (np.pi * self.maze.load.centroid_max_dist * self.theta_resolution) / (180 * self.pos_resolution)

    @property
    def theta_distance_factor(self):
        return (np.pi * self.maze.load.centroid_max_dist) / 180

    def _initialize_maze_edges(self):
        """
        set x&y edges to 0 in order to define the maze boundaries (helps the visualization)
        :return:
        """
        self.space[0, :, :] = 0
        self.space[-1, :, :] = 0
        self.space[:, 0, :] = 0
        self.space[:, -1, :] = 0

    def _maze_step(self, x, y, theta):
        self.maze.load.rotate(theta)
        self.maze.load.translate(x, y)
        if self.maze.load.intersects(self.maze.board):
            self.space[self.coords_to_indexes(x, y, theta)] = 0

    def calculate_space(self):
        # initialize 3d map for the phase_space
        self.space = np.ones((int(np.ceil(self.x_size / float(self.pos_resolution))),
                              int(np.ceil(self.y_size / float(self.pos_resolution))),
                              int(np.ceil(self.theta_size / float(self.theta_resolution)))))
        print("PhaseSpace: Calculating space")
        for theta in tqdm(np.arange(self.shape['theta'][0], self.shape['theta'][1], self.theta_resolution)):
            for x in np.arange(self.shape['x'][0], self.shape['x'][1], self.pos_resolution):
                for y in np.arange(self.shape['y'][0], self.shape['y'][1], self.pos_resolution):
                    self._maze_step(x, y, theta)

    def visualize_space(self):
        x, y, z = np.mgrid[self.shape['x'][0]:self.shape['x'][1]:self.pos_resolution,
                  self.shape['y'][0]:self.shape['y'][1]:self.pos_resolution,
                  self.shape['theta'][0] * self.theta_factor: self.shape['theta'][1] * self.theta_factor:complex(0,
                                                                                                                 self.theta_size / float(
                                                                                                                     self.theta_resolution))]
        mlab.figure()
        mlab.contour3d(x, y, z, self.space, opacity=0.7)
        # mlab.points3d(x, y, z, self.space_boundary)
        mlab.axes(xlabel="x", ylabel="y", zlabel="theta")
        mlab.orientation_axes(xlabel="x", ylabel="y", zlabel="theta")

    def iterate_neighbours(self, ix, iy, itheta):
        for dx, dy, dtheta in itertools.product([-1, 0, 1], repeat=3):
            _ix, _iy, _itheta = ix + dx, iy + dx, itheta + dtheta
            if ((_ix >= self.space.shape[0]) or (_ix < 0) or \
                    (_iy >= self.space.shape[1]) or (_iy < 0) or \
                    (_itheta >= self.space.shape[2]) or (_itheta < 0) or \
                    ((dx, dy, dtheta) == (0, 0, 0))):
                continue
            yield (_ix, _iy, _itheta)

    def plot_trajectory(self, traj, color=(0, 0, 0)):
        mlab.plot3d(traj[0],
                    traj[1],
                    traj[2] * self.theta_factor,
                    color=color, tube_radius=0.045, colormap='Spectral')
        mlab.points3d([traj[0, 0]], [traj[1, 0]], [traj[2, 0] * self.theta_factor])

    def save_space(self, path='SLT.pkl'):
        pickle.dump((self.space, self.space_boundary), open(path, 'wb'))

    def load_space(self, path='SLT.pkl'):
        (self.space, self.space_boundary) = pickle.load(open(path, 'rb'))
        # self.space = cPickle.load(file(path, 'rb'))

    def _is_boundary_cell(self, x, y, theta):
        if not self.space[x, y, theta]:
            return False
        for n_x, n_y, n_theta in self.iterate_neighbours(x, y, theta):
            if not self.space[n_x, n_y, n_theta]:
                return True
        return False

    def indexes_to_coords(self, ix, iy, itheta):
        return (self.shape['x'][0] + ix * self.pos_resolution,
                self.shape['y'][0] + iy * self.pos_resolution,
                self.shape['theta'][0] + itheta * self.theta_resolution)

    def coords_to_indexes(self, x, y, theta):
        return (int(round((x - self.shape['x'][0]) / self.pos_resolution)),
                int(round((y - self.shape['y'][0]) / self.pos_resolution)),
                int(round((theta - self.shape['theta'][0]) / self.theta_resolution)))

    def iterate_space_index(self):
        for ix in range(self.space.shape[0]):
            for iy in range(self.space.shape[1]):
                for itheta in range(self.space.shape[2]):
                    yield ix, iy, itheta

    def calculate_boundary(self):
        if self.space is None:
            self.calculate_space()
        self.space_boundary = np.zeros((int(np.ceil(self.x_size / float(self.pos_resolution))),
                                        int(np.ceil(self.y_size / float(self.pos_resolution))),
                                        int(np.ceil(self.theta_size / float(self.theta_resolution)))))
        for ix, iy, itheta in self.iterate_space_index():
            if self._is_boundary_cell(ix, iy, itheta):
                self.space_boundary[ix, iy, itheta] = 1

# TODO - mark traj begin point
