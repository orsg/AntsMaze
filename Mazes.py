from shapely import affinity, geometry
import matplotlib.pyplot as plt
import numpy as np

class Graph(object):

    def __init__(self, coords):
        self.shape = geometry.LineString(coords)

    def intersects(self, other):
        return self.shape.intersects(other.shape)

    def iterate_lines(self):
        for p1, p2 in zip(self.shape.coords[:-1], self.shape.coords[1:]):
            yield geometry.LineString((p1, p2))

    def visualize(self, ax=None):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        x, y = self.shape.xy
        ax.plot(x, y, alpha=0.7, linewidth=3, solid_capstyle='round', zorder=2)
        return ax

class Load(Graph):

    def __init__(self, coords):
        """
        :param coords:
        """
        super(Load, self).__init__(coords)
        self.centroid = self.shape.centroid.coords[:][0]
        self.theta = 0
        self.centroid_max_dist = self.calculate_max_distance_from_cetroid()

    def calculate_max_distance_from_cetroid(self):
        return geometry.Point(self.centroid).hausdorff_distance(self.shape)

    def translate(self, x, y):
        """
        move the center of mass to the given position
        :param x: new x in pixels
        :param y: new y in pixels
        :return:
        """
        self.shape = affinity.translate(self.shape,
                                        xoff=x-self.shape.centroid.x,
                                        yoff=y-self.shape.centroid.y)
        self.centroid = self.shape.centroid.coords[:][0]

    def rotate(self, theta):
        """
        rotate the object around the com to the given
        :theta phase: rotation theta in degrees
        :return:
        """
        if self.theta != theta:
            self.shape = affinity.rotate(self.shape, theta-self.theta, origin='centroid')
        self.theta=theta

    def visualize(self, ax=None):
        ax = super(Load, self).visualize(ax)
        ax.plot(self.centroid[0], self.centroid[1], 'ro')
        return ax

class Maze(object):

    def __init__(self, coords_board, coords_load, name=None):
        self.board = Graph(coords_board)
        self.load = Load(coords_load)
        self.name=name

    def visualize(self, ax=None):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        if self.name is not None:
            ax.set_title("Maze: " + self.name)
        self.board.visualize(ax)
        self.load.visualize(ax)
        ax.axis('scaled')
        ax.set_xlabel("x(cm)")
        ax.set_ylabel("y(cm)")
        plt.draw()
        return ax



def change_shape_ratio(shape, r):
    return [(x*r,y*r) for (x,y) in shape]

#According to Large dimensions
SLIT_S_SIZE = 0.495
SLIT_M_SIZE = 1.224
SLIT_L_SIZE = 2.448
SLIT_SL_SIZE = 3.66
SLIT_XL_SIZE = 4.896

BOARD_SLIT_HIEGHT = 15
BOARD_SLIT_WIDTH = 17.6

def get_slit_board(slit_size):
    slit_low = (BOARD_SLIT_HIEGHT - slit_size)/2
    slit_high = slit_low + slit_size
    x_edge_far = BOARD_SLIT_WIDTH + (slit_size) / 12.2
    return [(x_edge_far,0), (x_edge_far,slit_low), (BOARD_SLIT_WIDTH,slit_low), (BOARD_SLIT_WIDTH,0), (0,0), (0,BOARD_SLIT_HIEGHT), (BOARD_SLIT_WIDTH,BOARD_SLIT_HIEGHT),
              (BOARD_SLIT_WIDTH,slit_high), (x_edge_far, slit_high), (x_edge_far, BOARD_SLIT_HIEGHT)]

BOARD_SLIT_L = get_slit_board(SLIT_L_SIZE)
BOARD_SLIT_XL = get_slit_board(SLIT_XL_SIZE)
BOARD_SLIT_SL = get_slit_board(SLIT_SL_SIZE)
BOARD_SLIT_M = get_slit_board(SLIT_M_SIZE)
BOARD_SLIT_S = get_slit_board(SLIT_S_SIZE)

BOARD_SPECIAL_L = [(29.5, 19.05), (29.5, 11.35), (29.15, 11.35), (29.15, 19.05), (22.75, 19.05), (22.75, 11.35),
                   (22.4,11.35), (22.4,19.05), (0,19.05), (0,0), (22.4,0), (22.4,7.7), (22.75,7.7), (22.75,0),
                   (29.15,0), (29.15,7.7), (29.5,7.7), (29.5,0)]
BOARD_LONG_L = [(17.7, 0), (3, 0), (3, 3), (0, 3), (0, 15.6), (3, 15.6), (3, 18.62),
                (17.7, 18.62), (17.7, 14.38), (14.25,11.08), (13.6,11.08), (17.26,14.58), (17.26,18.62),
                (7.66,18.62), (7.66,10.19), (7.22, 10.19), (7.22, 12.52), (0, 12.52),
                (0,6.12), (7.22,6.12), (7.22, 8.43), (7.66,8.43), (7.66,0), (17.26,0), (17.26,4.04),
                (13.6,7.54), (14.25,7.54), (17.7,4.06), (17.7,0)]

LOAD_H_L = [(0, 0), (0.8, 0), (0.8, 1.0), (2.8, 1.0), (2.8, 0), (3.6, 0), (3.6, 2.8), (2.8, 2.8),
            (2.8,1.8), (0.8,1.8), (0.8,2.8), (0,2.8), (0,0)]
LOAD_T_L = [(2.8, 2.0), (2.8, 2.8), (0, 2.8), (0, 2.0), (1.0, 2.0), (1.0, 0), (1.8, 0),
            (1.8,2.0), (2.8,2.0)]

LOAD_T_XL = change_shape_ratio(LOAD_T_L, (SLIT_XL_SIZE / SLIT_L_SIZE))
LOAD_T_SL = change_shape_ratio(LOAD_T_L, (SLIT_SL_SIZE / SLIT_L_SIZE))
LOAD_T_M = change_shape_ratio(LOAD_T_L, (SLIT_M_SIZE / SLIT_L_SIZE))
LOAD_T_S = change_shape_ratio(LOAD_T_L, (SLIT_S_SIZE / SLIT_L_SIZE))

LOAD_I_L = [(0, 0), (1.0, 0), (1.0, 2.8), (0, 2.8), (0, 0)]
LOAD_ASYMMETRICAL_H_L = [(0, 0), (0.8, 0), (0.8, 2.22), (2.78, 2.22), (2.78, 1.22), (3.58, 1.22),
                         (3.58,5.24), (2.78,5.24), (2.78,3.02), (0.8,3.02), (0.8, 4.02), (0, 4.02), (0,0)]
LOAD_SPECIAL_L = [(0, 0), (0.8, 0), (0.8, 2.01), (8.83, 2.01), (8.83, 1.19), (9.63, 1.19),
                  (9.63,3.63), (8.83,3.63), (8.83,2.81), (0.8,2.81), (0.8, 4.82), (0, 4.82), (0,0)]
LOAD_LONG_L = [(0, 0), (0.44, 0), (0.44, 0.99), (8.03, 0.99), (8.03, 1.43),
               (0.44,1.43), (0.44,2.42), (0,2.42), (0,0)]

LOAD_H_XL = change_shape_ratio(LOAD_H_L, (SLIT_XL_SIZE / SLIT_L_SIZE))
LOAD_ASYMMETRICAL_H_XL = change_shape_ratio(LOAD_ASYMMETRICAL_H_L, (SLIT_XL_SIZE / SLIT_L_SIZE))
LOAD_I_XL = change_shape_ratio(LOAD_I_L, (SLIT_XL_SIZE / SLIT_L_SIZE))

MAZE_T_S = Maze(BOARD_SLIT_S, LOAD_T_S, "Slit T - Small")
MAZE_T_M = Maze(BOARD_SLIT_M, LOAD_T_M, "Slit T - Medium")
MAZE_T_L = Maze(BOARD_SLIT_L, LOAD_T_L, "Slit T - Large")
MAZE_T_SL = Maze(BOARD_SLIT_SL, LOAD_T_SL, "Slit T - Sesqui-Large")
MAZE_T_XL = Maze(BOARD_SLIT_XL, LOAD_T_XL, "Slit T - Extra-Large")
MAZE_I_L = Maze(BOARD_SLIT_L, LOAD_I_L, "Slit I - Large")
MAZE_H_L = Maze(BOARD_SLIT_L, LOAD_H_L, "Slit H - Large")
MAZE_H_XL = Maze(BOARD_SLIT_XL, LOAD_H_XL, "Slit H - Extra Large")
MAZE_I_XL = Maze(BOARD_SLIT_XL, LOAD_I_XL, "Slit I - Extra Large")
MAZE_ASYMMETRICAL_H_XL = Maze(BOARD_SLIT_XL, LOAD_ASYMMETRICAL_H_XL, "Slit Assymetrical H - Extra Large")


MAZE_LOAD_ASYMMETRICAL_H_L = Maze(BOARD_SLIT_L, LOAD_ASYMMETRICAL_H_L)
#
MAZE_SPECIAL = Maze(BOARD_SPECIAL_L, LOAD_SPECIAL_L)
MAZE_LONG = Maze(BOARD_LONG_L, LOAD_LONG_L)

BOARD_TEST = [(2,5),(0,5),(0,0),(5,0),(5,5),(3,5)]
LOAD_TEST = [(1,4),(1.5,4),(1.5,1),(1,1),(1,4)]
MAZE_TEST = Maze(BOARD_TEST, LOAD_TEST)
