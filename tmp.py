import scipy.io as sio
import numpy as np

DATA_PATH = '../../autodl-fs/houston2018/houston2018.mat'
data = sio.loadmat(DATA_PATH)

data['hsi'] = data['hsi'][1:49]

out = {
    'hsi': data['hsi'],
    'lidar': data['lidar'],
    'rgb': data['rgb'],
}
sio.savemat(DATA_PATH, out)