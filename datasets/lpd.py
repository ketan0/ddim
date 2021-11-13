#!/usr/bin/env python3

import pandas as pd
from torch.utils.data import Dataset
import glob
import numpy as np
import os
import pickle

class LakhPianoroll(Dataset):
    def __init__(self, data_dir):
        print('loading pianoroll paths...')
        fp = '/Users/ketanagrawal/CS236/final_proj/ddim/paths.pkl'
        if False:
            with open(fp, 'rb') as f:
                self.data_fps = pickle.load(f)
        else:
            self.data_fps = glob.glob(data_dir + "/*")
            with open(fp, 'wb') as f:
                pickle.dump(self.data_fps, f)

    def __len__(self):
        # return len(self.data_fps)
        return 100
    def __getitem__(self, index):
        pianoroll = np.load(self.data_fps[index])
        pianoroll = np.expand_dims(pianoroll, axis=0)
        return pianoroll, []
