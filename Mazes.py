from shapely import affinity, geometry
import matplotlib.pyplot as plt


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
        plt.show()
        return ax




def get_slit_board(slit_size):
    board_height=15
    x_edge = 17.6
    slit_low = (board_height - slit_size)/2
    slit_high = slit_low + slit_size
    x_edge_far = x_edge + (slit_size) / 12.2
    return [(x_edge_far,0), (x_edge_far,slit_low), (x_edge,slit_low), (x_edge,0), (0,0), (0,board_height), (x_edge,board_height),
              (x_edge,slit_high), (x_edge_far, slit_high), (x_edge_far, board_height)]

def change_shape_ratio(shape, r):
    return [(x*r,y*r) for (x,y) in shape]

#According to Large dimensions
SLIT_L_SIZE = 2.44
SLIT_SL_SIZE = 3.67
BOARD_SLIT_L = get_slit_board(SLIT_L_SIZE)
BOARD_SLIT_SL = get_slit_board(SLIT_SL_SIZE)
BOARD_SPECIAL = [(29.5, 19.05), (29.5,11.35), (29.15,11.35), (29.15, 19.05), (22.75,19.05), (22.75,11.35),
                 (22.4,11.35), (22.4,19.05), (0,19.05), (0,0), (22.4,0), (22.4,7.7), (22.75,7.7), (22.75,0),
                 (29.15,0), (29.15,7.7), (29.5,7.7), (29.5,0)]
BOARD_LONG = [(17.7,0),(3,0),(3,3),(0,3),(0,15.6),(3,15.6),(3,18.62),
              (17.7, 18.62), (17.7, 14.38), (14.25,11.08), (13.6,11.08), (17.26,14.58), (17.26,18.62),
              (7.66,18.62), (7.66,10.19), (7.22, 10.19), (7.22, 12.52), (0, 12.52),
            (0,6.12), (7.22,6.12), (7.22, 8.43), (7.66,8.43), (7.66,0), (17.26,0), (17.26,4.04),
              (13.6,7.54), (14.25,7.54), (17.7,4.06), (17.7,0)]

LOAD_H = [(0,0), (0.8,0), (0.8,1.0), (2.8,1.0), (2.8,0), (3.6,0), (3.6,2.8), (2.8,2.8),
          (2.8,1.8), (0.8,1.8), (0.8,2.8), (0,2.8), (0,0)]
LOAD_T = [(2.8,2.0), (2.8,2.8), (0,2.8), (0,2.0), (1.0, 2.0), (1.0,0), (1.8,0),
          (1.8,2.0), (2.8,2.0)]
LOAD_T_SL = change_shape_ratio(LOAD_T, (42.0/28.0))
LOAD_I = [(0,0), (1.0,0), (1.0,2.8), (0,2.8),(0,0)]
LOAD_ASYMMETRICAL_H = [(0,0),(0.8,0), (0.8,2.22),(2.78,2.22), (2.78,1.22), (3.58,1.22),
                       (3.58,5.24), (2.78,5.24), (2.78,3.02), (0.8,3.02), (0.8, 4.02), (0, 4.02), (0,0)]
LOAD_SPECIAL = [(0,0), (0.8,0), (0.8,2.01), (8.83,2.01), (8.83,1.19), (9.63,1.19),
                (9.63,3.63), (8.83,3.63), (8.83,2.81), (0.8,2.81), (0.8, 4.82), (0, 4.82), (0,0)]
LOAD_LONG = [(0,0), (0.44,0), (0.44,0.99), (8.03,0.99), (8.03,1.43),
             (0.44,1.43), (0.44,2.42), (0,2.42), (0,0)]

MAZE_T_L = Maze(BOARD_SLIT_L, LOAD_T, "Slit T - Large")
MAZE_T_SL = Maze(BOARD_SLIT_SL, LOAD_T_SL, "Slit T - Sesqui-Large")
MAZE_I_L = Maze(BOARD_SLIT_L, LOAD_I, "Slit I - Large")
MAZE_H_L = Maze(BOARD_SLIT_L, LOAD_H, "Slit H - Large")
MAZE_LOAD_ASYMMETRICAL_H_L = Maze(BOARD_SLIT_L, LOAD_ASYMMETRICAL_H)
#
MAZE_SPECIAL = Maze(BOARD_SPECIAL, LOAD_SPECIAL)
MAZE_LONG = Maze(BOARD_LONG, LOAD_LONG)

BOARD_TEST = [(2,5),(0,5),(0,0),(5,0),(5,5),(3,5)]
LOAD_TEST = [(1,4),(1.5,4),(1.5,1),(1,1),(1,4)]
MAZE_TEST = Maze(BOARD_TEST, LOAD_TEST)

# MAZE_SPECIAL.visualize()
# MAZE_T_L.visualize()
# MAZE_T_SL.visualize()