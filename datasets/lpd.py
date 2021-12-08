#!/usr/bin/env python3

# import pandas as pd
from torch.utils.data import Dataset
import glob
import numpy as np
# import os
# import pickle

class LakhPianoroll(Dataset):
    def __init__(self, data_dir):
        print('loading pianoroll paths...')
        # fp = '/Users/ketanagrawal/cs236_final_proj/ddim/paths.pkl'
        # with open(fp, 'rb') as f:
        #     self.data_fps = pickle.load(f)
        self.data_fps = glob.glob(data_dir + '/*')
        # with open(fp, 'wb') as f:
        #     pickle.dump(self.data_fps, f)

    def __len__(self):
        return len(self.data_fps)

    def __getitem__(self, index):
        pianoroll = np.load(self.data_fps[index])
        pianoroll = np.expand_dims(pianoroll, axis=0)
        return pianoroll, []

class LakhPianorollEmbed(Dataset):
    def __init__(self, x_fp: str):
        self.x = np.load(x_fp)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x_idx = self.x[index]
        # x_idx = np.expand_dims(x_idx, axis=0)
        return x_idx, []
