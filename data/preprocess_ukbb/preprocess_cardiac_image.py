'''
* Licensed under the Apache License, Version 2.
* By Siyi Du, 2024

Create cardiac image (3,210,210) from sa_ED, sa, sa_ES, and save as npy
Usage:
nohup python -u create_cardiac_task_image.py > out_npy_preproceed.log 2>&1 &
'''
import os
import csv
from os.path import join
import re
import random
import glob

import numpy as np
import torch
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import seaborn as sns
import nibabel as nib
from matplotlib import pyplot as plt
import matplotlib_venn
from torchvision import transforms
from torchvision.ops import masks_to_boxes
from torchvision.transforms.functional import crop
from sklearn.model_selection import train_test_split

from typing import List, Union
import operator
from tqdm import tqdm

from Utils import check_or_save

# Read DICOM header
from pydicom.filereader import dcmread
from os import listdir
from os.path import isfile, join
import multiprocessing
from multiprocessing import Pool
import time
from glob import glob


def power(tensor, gamma):
    if tensor.min() < 0:
        output = tensor.sign() * tensor.abs() ** gamma
    else:
        output = tensor ** gamma
    return output


class RandomGamma(torch.nn.Module):
    def __call__(self, pic):
        ran = np.random.uniform(low=0.25,high=1.75)
        transformed_tensors = power(pic,ran)
        return transformed_tensors
    
    def __repr__(self):
        return self.__class__.__name__ + '()'
    
    def power(tensor, gamma):
        if tensor.min() < 0:
            output = tensor.sign() * tensor.abs() ** gamma
        else:
            output = tensor ** gamma
        return output


# BASE = '/vol/biomedic3/sd1523/data/mm/UKBB'
BASE = '/bigdata/siyi/data/UKBB'
image_base_folder = '/vol/biodata/data/biobank/18545/data/*'
RAWDATA = '/vol/biodata/data/biobank/18545/data'


FEATURES = join(BASE,'features')
DATADICT = join(FEATURES,'Data_Dictionary_Showcase.csv')
CLEAN_FEATURES = join(FEATURES,'cardiac_features_18545_clean.csv')


SEGMENTATIONS = join(BASE,'cardiac_segmentations')
PROJECT_DATA = join(SEGMENTATIONS,'projects','SelfSuperBio', '18545')
SUBJECT_DATA = join(SEGMENTATIONS,'subjects')
VECTOR_FEATURES = join(PROJECT_DATA,'cardiac_feature_18545_vector_labeled_noOH_dropNI.csv')


def get_mid_beat_slice(im, es_slice):
    # thresh=(1.0, 99.0)
    best_overlap_es = 0
    for i in range(50):
        im_slice = im[:,:,im.shape[2]//2,i]
        overlap_es = (es_slice==im_slice).sum()
        if overlap_es > best_overlap_es:
            best_overlap_es = overlap_es
            best_i_es = i

    # val_l, val_h = np.percentile(im, thresh)
    im_slice = im[:,:,im.shape[2]//2,best_i_es]
    # im_slice[im_slice > val_h] = val_h
    try:
        assert np.allclose(im_slice,es_slice)
        match = True
    except:
        print((im_slice==es_slice).sum()/im_slice.size)
        match = False
    mid_beat_i = best_i_es//2
    mid_beat_slice = im[:,:,im.shape[2]//2,mid_beat_i]
    # mid_beat_slice[mid_beat_slice > val_h] = val_h
    return mid_beat_slice, match


def create_sa_es_ed_mm(_id):
    '''Load ED, ES, and one frame between ED and ES, choose the middle slice of these three frames,
        pad to square, stack them, and save the numpy array  (3,210,210)'''
    to_stack = []
    problem_id = None
    missing_id = None
    not_matching_id = None
    ims_stacked_t_n = None
    save_np_path = None
    for cycle_position in ['sa_ES.nii.gz', 'sa.nii.gz', 'sa_ED.nii.gz']:
        path = join(RAWDATA, str(_id), cycle_position)
        if os.path.exists(path):
            nii = nib.load(path)
            im = nii.get_fdata()
        else:
            print('Missing files:', path)
            missing_id=_id
            break

        # Too few z-axis slices are bad quality images
        # if im.shape[2] <= 7:
            # print(f'Too few z-axis slices: {path}')
            # break
            
        # Full cycle volumes are used to extract middle of heart beat slice
        if cycle_position == 'sa.nii.gz':
            mid_heart_slice, match = get_mid_beat_slice(im, es_slice)
            if not match:
                not_matching_id=_id
                print(f'Not matching ES: {path}')
        else:
            mid_heart_slice = im[:,:,im.shape[2]//2]

        # Set es_slice to be used during extraction of mid beat
        if cycle_position == 'sa_ES.nii.gz':
            es_slice = mid_heart_slice

        # pad to square 
        if mid_heart_slice.shape[1]>mid_heart_slice.shape[0]:
            mid_heart_slice = np.pad(mid_heart_slice, ((((mid_heart_slice.shape[1]-mid_heart_slice.shape[0])//2), ((mid_heart_slice.shape[1]-mid_heart_slice.shape[0])//2)), (0, 0)), 'constant', constant_values=0)
        else:
            mid_heart_slice = np.pad(mid_heart_slice, ((0, 0), (((mid_heart_slice.shape[0]-mid_heart_slice.shape[1])//2), ((mid_heart_slice.shape[0]-mid_heart_slice.shape[1])//2))), 'constant', constant_values=0)
        try:
            assert mid_heart_slice.shape[0]==mid_heart_slice.shape[1], print(mid_heart_slice.shape[0], mid_heart_slice.shape[1])
        except:
            print(f'Shapes didnt match: {path}')
            break
        to_stack.append(mid_heart_slice)

    if len(to_stack) == 3:
        ims_stacked_t_n = np.stack(to_stack, axis=0)
        w = ims_stacked_t_n.shape[1]
        if w % 2 != 0:
            pad = ((210-w)//2, (210-w)//2+1)
        else:
            pad = ((210-w)//2, (210-w)//2)
        ims_stacked_t_n = np.pad(ims_stacked_t_n, ((0,0), (pad[0],pad[1]), (pad[0],pad[1])), 'constant', constant_values=0)
        assert ims_stacked_t_n.shape == (3, 210, 210), print(ims_stacked_t_n.shape)
        os.makedirs(join(SUBJECT_DATA, str(_id)), exist_ok=True)
        ims_stacked_t_n = ims_stacked_t_n.astype(np.float32)
        ims_stacked_t_n = ims_stacked_t_n.permute(1,2,0)
        ims_stacked_t_n = ims_stacked_t_n / np.max(ims_stacked_t_n, axis=(0,1), keepdims=True)
        assert ims_stacked_t_n.shape == (210, 210, 3)
        save_np_path = join(SUBJECT_DATA, str(_id), f'sa_es_ed_mm.npy')
        np.save(save_np_path, ims_stacked_t_n)
        ims_stacked_t_n = torch.from_numpy(ims_stacked_t_n)
        # all_subejects[_id] = ims_stacked_t_n
        # all_npy_path[_id] = save_np_path
    else:
        problem_id=_id
    print(f'Completed: {_id}')
    return (_id, ims_stacked_t_n, save_np_path, missing_id, not_matching_id, problem_id)


def preproceed_img(path):
    _id = int(path.split('/')[-2])
    img = np.load(path)
    img = img.astype(np.float32)
    if img.shape == (3,210,210):
        img = img.transpose(1,2,0)
    elif img.shape == (210,210,3):
        pass
    assert img.max() > 1.0
    img = img/np.max(img,axis=(0,1),keepdims=True)

    assert img.shape == (210,210,3)
    assert img.dtype == np.float32
    assert img.max() <= 1.0
    print(f'Completed: {_id}')
    # np.save(f'{_id}.npy', img)
    np.save(path, img)



if __name__ == '__main__':
    SAVE = False
    if SAVE:
        datadict_df = pd.read_csv(DATADICT,quotechar='"',escapechar='\\')
        # There are two BMI fields with the same name. One is measured by impedance though (instead of the standard way) and thus gets a different name
        datadict_df.loc[datadict_df['FieldID']==23104,'Field']='Body mass index (BMI) Impedance'

        data_df=pd.read_csv(VECTOR_FEATURES)
        print(f'Num of subjects in tabular: {len(data_df)}')
        _ids = list(data_df['eid'].astype(int))
        # _ids = _ids[:5400]

        # id: npy path
        all_npy_path = {}
        # don't have sa_ES or sa_ED
        problem_ids = []
        missing_ids = []
        not_matching_ids = []
        
        for i in range(len(_ids)//5000+1):
            start_time = time.time()
            start, end = i*5000, min((i+1)*5000, len(_ids))
            all_subjects = {}
            print(f'Start {i}th step: {start} to {end}')
            pool = multiprocessing.Pool(processes=20)
            results = pool.map(create_sa_es_ed_mm, _ids[start:end])
            pool.close()
            pool.join()
            for result in tqdm(results):
                _id, ims_stacked_t_n, save_np_path, missing_id, not_matching_id, problem_id = result
                if ims_stacked_t_n is not None:
                    all_subjects[_id] = ims_stacked_t_n
                if save_np_path is not None:
                    all_npy_path[_id] = save_np_path
                if missing_id is not None:
                    missing_ids.append(missing_id)
                if not_matching_id is not None:
                    not_matching_ids.append(not_matching_id)
                if problem_id is not None:
                    problem_ids.append(problem_id)
            # TODO uncomment this if you want to save all the subjects into several pt files
            # torch.save(all_subjects, join(PROJECT_DATA, f'preprocessed_cardiac_dict_{i}.pt'))
            time.sleep(10)
            end_time = time.time()
            time_elapsed = end_time-start_time
            print('Finished {}th step complete in {:.0f}m {:.0f}s'.
                format(i, time_elapsed // 60, time_elapsed % 60))

        print(f'Num of problem: {len(problem_ids)}, Num of not matching: {len(not_matching_ids)}, Num of missing {len(missing_ids)}')

        
        torch.save(all_npy_path, join(PROJECT_DATA, 'preprocessed_cardiac_npy_path.pt'))
        torch.save(problem_ids, join(PROJECT_DATA, 'problem_ids_cardiac.pt'))
        torch.save(not_matching_ids, join(PROJECT_DATA, 'not_matching_ids_cardiac.pt'))
        torch.save(missing_ids, join(PROJECT_DATA, 'missing_ids_cardiac.pt'))

    # TODO uncomment this if you want to merge all the pt files into one
    # glob_str=join(PROJECT_DATA,'preprocessed_cardiac_dict_*.pt')
    # files = glob(glob_str)
    # files.sort()
    # print(files)
    # merged_subjects = {}
    # for file in tqdm(files):
    #     subjects = torch.load(file)
    #     merged_subjects.update(subjects)
    # torch.save(merged_subjects, join(PROJECT_DATA, 'preprocessed_cardiac_dict.pt'))



    all_npy_paths = torch.load(join(PROJECT_DATA, 'preprocessed_cardiac_npy_path.pt'))
    all_npy_paths = list(all_npy_paths.values())
    pool = multiprocessing.Pool(processes=40)
    pool.map(preproceed_img, all_npy_paths)
    pool.close()
    pool.join()
