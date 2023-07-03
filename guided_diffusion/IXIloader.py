import os
import sys
import pickle
import cv2
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import pandas as pd
from skimage.transform import rotate
import torch
import torch.nn
import numpy as np
import os
import os.path
import nibabel

class IXIDataset(torch.utils.data.Dataset):
    def __init__(self, directory, transform, test_flag=True):
        '''
        directory is expected to contain some folder structure:
                  if some subfolder contains only files, all of these
                  files are assumed to have a name like
                  IXI_train_001_XXX_123_w.nii.gz
                  where XXX is one of t1, t2, PD
                  we assume these four files belong to the same image
                  pveseg is supposed to be the tissue region masks segmented by FAST
        '''
        super().__init__()
        self.directory = os.path.expanduser(directory)
        self.transform = transform

        # generate the 'pveseg' region masks
        image_path = self.directory
        modality = 't2'
        floaders = os.listdir(image_path)
        for floader in floaders:
            niifiles = os.listdir(image_path + '/' + floader)
            filedic = [image_path + '/' + floader + '/' + niifile for niifile in niifiles]
            for niifile in niifiles:
                if (niifile.split('_')[-1] == modality + '.nii.gz'):
                    cmd = 'fast -s 1 -t 2 -n 5 -H 0.1 -I 4 -l 20.0 -o ' + image_path + '/' + floader + '/' + niifile
                    os.system(cmd)
                    # print(niifile)

        self.test_flag = test_flag
        if test_flag:
            self.seqtypes = ['t1', 'PD',  'pveseg']
        else:
            self.seqtypes = ['t1', 'PD', 'pveseg', 't2']

        self.seqtypes_set = set(self.seqtypes)
        self.database = []
        path1 = os.listdir(self.directory)
        for path in path1:
            for root, dirs, files in os.walk(self.directory + "/" + path):
                # if there are no subdirs, we have data
                if not dirs:
                    files.sort()
                    datapoint = dict()
                    # extract all files as channels
                    for f in files:
                        seqtype = f.split('_')[4]  # the 3th string in nii.gz file,like t1\t2
                        datapoint[seqtype] = os.path.join(root, f)
                    assert set(datapoint.keys()) == self.seqtypes_set, \
                        f'datapoint is incomplete, keys are {datapoint.keys()}'
                    self.database.append(datapoint)

    def __getitem__(self, x):
        out = []
        filedict = self.database[x]
        for seqtype in self.seqtypes:
            nib_img = nibabel.load(filedict[seqtype])
            path = filedict[seqtype]
            out.append(torch.tensor(nib_img.get_fdata()))
        out = torch.stack(out)
        if self.test_flag:
            image = out
            image = image[..., 16:-16, 16:-16]  # crop to a size of (224, 224)
            if self.transform:
                image = self.transform(image)
            return (image, image, path)
        else:

            image = out[:-1, ...]
            label = out[-1, ...][None, ...]
            image = image[..., 16:-16, 16:-16]  # crop to a size of (224, 224)
            label = label[..., 16:-16, 16:-16]
            # label=torch.where(label > 0, 1, 0).float()  #merge all tumor classes into one
            if self.transform:
                state = torch.get_rng_state()
                image = self.transform(image)
                torch.set_rng_state(state)
                label = self.transform(label)
                # print(label)
                # print(image)
            return (image, label)

    def __len__(self):
        return len(self.database)