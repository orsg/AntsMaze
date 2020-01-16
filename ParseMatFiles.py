from scipy.io import loadmat
import glob
import pickle
import os
import numpy as np

dir = r'C:\Users\orsa\Downloads\wetransfer-59af8b'
out_dir = r'C:\Users\orsa\PycharmProjects\AntsMaze\trajectories'
paths = glob.glob("{}\*.mat".format(dir))
def parse():
    for p in paths:
        matfile = loadmat(p)
        load_center = matfile['load_center']
        load_orientation = matfile['shape_orientation']
        pickle.dump((load_center, load_orientation), open(os.path.join(out_dir, os.path.basename(p).replace('mat', 'pkl')), 'wb'))

pairs = [("4080014_2","4080015_1"),
        ("4090002_10", "4090003_1"),
        ("4090004_13", "4090005_1"),
        ("4100002_14", "4100003_1")]

def create_pairs():
    for n1,n2 in pairs:
        p1 = [p for p in paths if n1 in p][0]
        p2 = [p for p in paths if n2 in p][0]
        mat1, mat2 = loadmat(p1), loadmat(p2)
        load_center = np.concatenate([mat1['load_center'], mat2['load_center']], axis=0)
        load_orientation = np.concatenate([mat1['shape_orientation'], mat2['shape_orientation']], axis=0)
        pickle.dump((load_center, load_orientation), open(os.path.join(out_dir, os.path.basename(p1).replace('mat', 'pkl')), 'wb'))

create_pairs()