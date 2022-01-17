# Wearable sensor dataset.

import os
import copy
import numpy as np
import pandas as pd
from PIL import Image
from os.path import join
from itertools import chain
from collections import defaultdict

from wfdb import processing
from scipy.signal import decimate, resample

from scipy.io import loadmat
import torch
import torch.utils.data as data
from torchaudio.transforms import Spectrogram

import nlpaug.augmenter.spectrogram as nas
import nlpaug.flow as naf
import re
import wfdb
import ast
from src.datasets.root_paths import DATA_ROOTS
DIAGNOSTIC_SUBCLASS=['270492004', '164889003', '164890007', '426627000', '713427006', 
                  '713426002', '445118002', '39732003', '164909002', '251146004', 
                  '698252002', '10370003', '284470004', '427172004', '164947007', 
                  '111975006', '164917005', '47665007', '427393009',
                  '426177001', '426783006', '427084000', '164934002',
                  '59931005']
EQUIV_CLASS = {
    '59118001':'713427006',
    '63593006':'284470004',
    '17338001':'427172004'
}



FEATURE_MEANS=np.array([-0.00074703,  0.00054328,  0.00128943,  0.0001024 , -0.00096791,
        0.00094267,  0.0008255 , -0.00062468, -0.00335543, -0.00189922,
        0.00095845,  0.000759  ])

FEATURE_STDS=np.array([0.13347071, 0.19802795, 0.15897414, 0.14904783, 0.10836737,
       0.16655428, 0.17850298, 0.33520913, 0.28028072, 0.27132468,
       0.23750131, 0.19444742])


class PTB_XL_FULL(data.Dataset):
    NUM_CLASSES = 24  # NOTE: They're not contiguous labels.
    NUM_CHANNELS = 12 # Multiple sensor readings from different parts of the body
    FILTER_SIZE = 32
    MULTI_LABEL = False

    def __init__(
        self,
        mode='train',
        sensor_transforms=None,
        root=DATA_ROOTS['ptb_xl_full'],
        examples_per_epoch=10000  # Examples are generated stochastically.
    ):
        super().__init__()
        self.examples_per_epoch = examples_per_epoch
        self.sensor_transforms = sensor_transforms
        self.dataset = BasePTB_XL_FULL(
            mode=mode, 
            root=root, 
            examples_per_epoch=examples_per_epoch)
    
    def transform(self, spectrogram):
        if self.sensor_transforms:
            if self.sensor_transforms == 'spectral':
                spectral_transforms = SpectrumAugmentation()
            elif self.sensor_transforms == 'spectral_noise':
                spectral_transforms = SpectrumAugmentation(noise=True)
            elif self.sensor_transforms == 'just_time':
                spectral_transforms = SpectrumAugmentation(just_time=True)
            else:
                raise ValueError(f'Transforms {self.sensor_transforms} not implemented.')

            spectrogram = spectrogram.numpy().transpose(1, 2, 0)
            spectrogram = spectral_transforms(spectrogram)
            spectrogram = torch.tensor(spectrogram.transpose(2, 0, 1))
        elif self.sensor_transforms:
            raise ValueError(
                f'Transforms "{self.sensor_transforms}" not implemented.')
        return spectrogram

    def __getitem__(self, index):
        # pick random number
        img_data, label = self.dataset.__getitem__(index)
        subject_data = [
            index,
            self.transform(img_data).float(), 
            self.transform(img_data).float(),
            label]

        return tuple(subject_data)

    def __len__(self):
        return self.examples_per_epoch



class BasePTB_XL_FULL(data.Dataset):

    def __init__(
        self,
        mode='train',
        root=DATA_ROOTS['ptb_xl_full'],
        measurements_per_example=1000,
        examples_per_epoch=10000,
        normalize=True
    ):
        super().__init__()
        self.examples_per_epoch = examples_per_epoch
        self.measurements_per_example = measurements_per_example  # Measurements used to make spectrogram
        self.mode = mode
        self.subject_data = self.load_data(root)
        self.normalize = normalize

    def get_subject_ids(self, mode):
        if mode == 'train':
            nums = [0, 1, 4, 5, 6, 7, 8, 9]
        elif mode == 'train_small':
            nums = [1]
        elif mode == 'val':
            nums = [2]
        elif mode == 'test':
            nums = [3]
        else:
            raise ValueError(f'mode must be one of [train, train_small, val, test]. got {mode}.')
        return nums  

    def load_data(self, root_path):
        path = r"/home/chw5444/vmk/viewmaker/data/ptb_xl_full/raw/"
        fs=500
        # # load and convert annotation data
        # Y = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
        Y = pd.read_csv(os.path.join(path,'splits_sub.csv')

        def load_raw_data(df, fs, path):
            df['filename_hr']= df.Patient.apply( lambda x: str(path+"/"+(x + '.hea')))
            data=[]
            lbls=[]
            for header_file in df.filename_hr:
                with open(header_file, 'r') as f:
                    header = f.readlines()    
                sampling_rate = int(header[0].split()[2])    
                mat_file = header_file.replace('.hea', '.mat')
                x = loadmat(mat_file)
                mat_file = header_file.replace('.hea', '')
                x1 = wfdb.rdsamp(mat_file)   

                lbl=re.findall(r'\b\d+\b',x1[1]['comments'][2])[0]
                recording = np.asarray(x['val'], dtype=np.float64)
                # Standardize sampling rate
                if sampling_rate > fs:
                    recording = decimate(recording, int(sampling_rate / fs))
                elif sampling_rate < fs:
                    recording = resample(recording, int(recording.shape[-1] * (fs / sampling_rate)), axis=1)
                data.append(recording.T[:5000,:])
                lbls.append(lbl)
            return np.stack(data),np.array(lbls)
        # Split data into train and test
        nums= self.get_subject_ids(self.mode)
        # Train
        X ,y= load_raw_data(Y, fs, path)
        X_train = X[np.where(Y.fold.isin(nums))]
        #         print("X train shape:", X_train.shape)
        y_train = y[np.where(Y.fold.isin(nums))]
        #         print("y data", y_train)
        #         print("y train test", y_train[0][0])
        #         print("y train shape:", y_train.shape)
        #         print()
        subject_data=[X_train,y_train]
        return subject_data
    
    def __getitem__(self, index):
        while True:
            ecgid = np.random.randint(len(self.subject_data[0]))
            if len(self.subject_data[1][ecgid]) > 0: break
                
#         print("example diagnosis id", self.subject_data[1][ecgid])
        
#         max_conf=np.argmax(self.subject_data[2][ecgid])
#         diagnosis_id = DIAGNOSTIC_SUBCLASS.index(self.subject_data[1][ecgid][max_conf])
        measurements = self.subject_data[0][ecgid]
        diagnosis_id = self.subject_data[1][ecgid]
        # Yields spectrograms of shape [52, 32, 32]
        spectrogram_transform=Spectrogram(n_fft=64-1, hop_length=32, power=2)
        spectrogram = spectrogram_transform(torch.tensor(measurements.T))
        spectrogram = (spectrogram + 1e-6).log()
        if self.normalize:
            spectrogram = (spectrogram - FEATURE_MEANS.reshape(-1, 1, 1)) / FEATURE_STDS.reshape(-1, 1, 1)
#         print("spectrogram shape", spectrogram.shape)
#         print("diagnosis_id", diagnosis_id)
        if diagnosis_id in EQUIV_CLASS.keys():
            diagnosis_id=EQUIV_CLASS[diagnosis_id]
        return spectrogram, diagnosis_id

    
    def __len__(self):
        return self.examples_per_epoch


class SpectrumAugmentation(object):

    def __init__(self, just_time=False, noise=False):
        super().__init__()
        self.just_time = just_time
        self.noise = noise

    def get_random_freq_mask(self):
        return nas.FrequencyMaskingAug(mask_factor=20)

    def get_random_time_mask(self):
        return nas.TimeMaskingAug(coverage=0.7)

    def __call__(self, data):
        if self.just_time:
            transforms = naf.Sequential([self.get_random_time_mask()])
        else: 
            transforms = naf.Sequential([self.get_random_freq_mask(),
                                     self.get_random_time_mask()])
        data = transforms.augment(data)
        if self.noise:
            noise_stdev = 0.25 * np.array(FEATURE_STDS).reshape(1, 1, -1)
            noise = np.random.normal(size=data.shape) * noise_stdev
            data = data + noise
        return data
