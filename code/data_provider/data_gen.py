import os, sys
from scipy.io import loadmat
import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.model_selection import train_test_split
import random
import tifffile as tiff
import math
import matplotlib.pyplot as plt


DATA_PATH_PREFIX = "../../data"
NOISE_DATA_SAVE_PREFIX = "../../data/noise_data"
NOISE_DATA_SAVE_PREFIX_DATA = "%s/data" % NOISE_DATA_SAVE_PREFIX
NOISE_DATA_SAVE_PREFIX_IMG= "%s/img" % NOISE_DATA_SAVE_PREFIX

def load_data(data_sign, data_path_prefix):
    if data_sign == "Indian":
        data = sio.loadmat('%s/Indian_pines_corrected.mat' % data_path_prefix)['indian_pines_corrected']
        labels = sio.loadmat('%s/Indian_pines_gt.mat' % data_path_prefix)['indian_pines_gt']
    elif data_sign == "Pavia":
        data = sio.loadmat('%s/PaviaU.mat' % data_path_prefix)['paviaU']
        labels = sio.loadmat('%s/PaviaU_gt.mat' % data_path_prefix)['paviaU_gt'] 
    elif data_sign == "Houston":
        data = sio.loadmat('%s/Houston.mat' % data_path_prefix)['img']
        labels = sio.loadmat('%s/Houston_gt.mat' % data_path_prefix)['Houston_gt']
    elif data_sign == 'Salinas':
        data = sio.loadmat('%s/Salinas_corrected.mat' % data_path_prefix)['salinas_corrected']
        labels = sio.loadmat('%s/Salinas_gt.mat' % data_path_prefix)['salinas_gt']
    elif data_sign == 'WH' or data_sign=='Honghu':
        data = sio.loadmat('%s/WHU_Hi_HongHu.mat' % data_path_prefix)['WHU_Hi_HongHu']
        labels = sio.loadmat('%s/WHU_Hi_HongHu_gt.mat' % data_path_prefix)['WHU_Hi_HongHu_gt']
    return data, labels


def norm_0_255(data):
    h, w, c = data.shape
    res = np.zeros_like(data)
    params = np.zeros([c,2])
    for ci in range(c):
        ss = data[:,:,ci]
        res[:,:,ci] = (ss - ss.min()) / (ss.max() - ss.min()) * 255
        params[ci] = np.asarray([ss.min(), ss.max()])
    res = res.astype(np.uint8)
    
    return res


class DataNoiseGeneratorBase(object):
    def __init__(self, data_sign) -> None:
        self.noise_type = "Base"
        self.data_sign = data_sign

        self.params = {}

    
    def gen(self, data):
        return data


    def save_data(self, data, path_prefix):
        if not os.path.exists(path_prefix):
            os.makedirs(path_prefix)

        res = {
            "data" : data,
            "params": self.params
        }

        path = "%s/%s_%s.mat" % (path_prefix, self.noise_type, self.data_sign)
        sio.savemat(path, res)

    def plot_img(self, data, path_prefix):
        assert len(data.shape) == 3
        h, w, c = data.shape
        assert c >= 3
        img = norm_0_255(data[:,:,[0,c//2,c-1]])

        path = "%s/%s_%s.jpg" % (path_prefix, self.noise_type, self.data_sign)
        if not os.path.exists(path_prefix):
            os.makedirs(path_prefix)
        plt.figure(figsize=(15,10))
        plt.imsave(path, img)

        

class JPEGGenerator(DataNoiseGeneratorBase):
    def __init__(self, data_sign) -> None:
        super().__init__(data_sign)

        self.noise_type = "jpeg"
        self.params = {
            'compression_ratios' : [10]
        }


    def gen(self, data):
        import glymur
        compression_ratios = self.params['compression_ratios']
        temp_file = 'output_lossy.jp2'
        jp2k = glymur.Jp2k(temp_file, data, cratios=compression_ratios)
        res_data = jp2k[:]
        return res_data


kvs = {
    'jpeg': JPEGGenerator,
}

def run_gen(data_sign):
    data, labels = load_data(data_sign, data_path_prefix=DATA_PATH_PREFIX)
    # start to gen
    for k, cls in kvs.items():
        obj = cls(data_sign)
        print("start to gen noise data_sign=%s, noise_type=%s .." % (data_sign, obj.noise_type))
        res_data = obj.gen(data) 
        obj.save_data(res_data, NOISE_DATA_SAVE_PREFIX_DATA)
        obj.plot_img(res_data, NOISE_DATA_SAVE_PREFIX_IMG)
        print("start to gen noise data_sign=%s, noise_type=%s .." % (data_sign, obj.noise_type))



def run_all():
    data_signs = ['Indian', 'Pavia'] 
    for data_sign in data_signs:
        run_gen(data_sign)



if __name__ == "__main__":
    run_all()

