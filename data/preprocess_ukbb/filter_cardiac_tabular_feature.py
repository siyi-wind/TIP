'''
* Licensed under the Apache License, Version 2.
* By Siyi Du, 2024

- Load all tabular features from UKBB
- Filter cardiac disease related tabular features.
- Save the filtered tabular features as csv (CARDIAC_FEATURES_PATH, CARDIAC_PATIENTS_PATH)
'''

import os
import csv
from os.path import join
import re
import random
import multiprocessing as mp
from glob import glob

import numpy as np
import torch
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import seaborn as sns
import nibabel as nib
from matplotlib import pyplot as plt
from torchvision import transforms
from tqdm import tqdm

from typing import List, Union
import operator

import sys
sys.path.append("../../")
# from utils.tabular_utils import *
from tabular_utils import *


BASE_PATH = '/vol/biomedic3/sd1523/data/mm/UKBB/features'
SUBJECT_DATA =  '/vol/biodata/data/biobank/18545/data'                            
RAW_DATA_PATH = '/vol/biodata/data/biobank/18545/downloaded'      
EXTRACTED_DATA_PATH = ''
DATA_PATH = join(RAW_DATA_PATH,'phenotype.csv')
CARDIAC_FEATURES_PATH = join(BASE_PATH,'cardiac_features_18545.csv')
CARDIAC_PATIENTS_PATH = join(BASE_PATH,'cardiac_features_18545_imaging.csv')
DATADICT_PATH = join(BASE_PATH,'Data_Dictionary_Showcase.csv')
BRIDGE_PATH = join(BASE_PATH,'Bridge_eids_60520_87802.csv')




datadict_df = pd.read_csv(DATADICT_PATH,quotechar='"',escapechar='\\')
# There are two BMI fields with the same name. One is measured by impedance though (instead of the standard way) and thus geets a different name
datadict_df.loc[datadict_df['FieldID']==23104,'Field']='Body mass index (BMI) Impedance'

# Need to use pandas Int64 to represent integers because missing values are floats normally in pandas
datatype_dict = {'Integer':"Int64", 'Categorical single':object, 'Date':str, 'Text':str, 'Continuous':float,
       'Time':str, 'Compound':object, 'Categorical multiple':object}

dtype = {}
dates = []
field_id2name = {}
for indx, row in datadict_df.iterrows():
    baseID = row['FieldID']
    instances = row['Instances']
    array = row['Array']
    field_id2name[baseID] = row['Field']
    for instance in range(instances):
        for arr in range(array):
            ID = '{}-{}.{}'.format(baseID,instance,arr)
            value_type = row['ValueType']
            if value_type == 'Time' or value_type == 'Date':
                dates.append(ID)
            dt = datatype_dict[value_type]
            if baseID==46:
                dt = float
            dtype[ID] = dt

print(len(dtype))

def read_csv(filename):
        return pd.read_csv(filename,header=None)

def multithread_read(glob_str: str) -> pd.DataFrame:
        files = glob(glob_str)
        files.sort()
        print(files)
        threads = len(files)
        with mp.Pool(processes=threads) as pool:
                df_list = pool.map(read_csv,files)
        final_frame = pd.concat(df_list,ignore_index=True)
                
        print (f"There are {len(final_frame)} rows of data")
        return final_frame

def multi_merge(glob_str: str) -> pd.DataFrame:
        files = glob(glob_str)
        files.sort()
        print(files)
        merged_df = pd.read_csv(files[0])
        print(len(merged_df), files[0])
        for file in tqdm(files[1:]):
                df = pd.read_csv(file)
                print(len(df), file)
                merged_df = pd.merge(merged_df,df,on='eid')
        print (f"There are {len(merged_df)} rows of data")
        print (f"There are {len(merged_df.columns)} columns of data")
        return merged_df


data_df = pd.read_csv(DATA_PATH, dtype=dtype, nrows=5)


cardiac_features = ['49', '21001', '12675', '12144', '874', '12338', '904', '20116', '1001', '20406', '50', '12697', '20415', '22425', '20421', '2634', '42008', '1349', '3894', '22334', '3627', '2188', '22508', '1279', '22330', '2966', '22432', '120007', '1379', '1299', '22426', '4079', '6164', '22506', '22507', '23283', '20162', '22410', '981', '23100', '22409', '22434', '12671', '1239', '93', '20428', '12674', '924', '2296', '12684', '22331', '20549', '1021', '22424', '22332', '4717', '12687', '2306', '1160', '12336', '12688', '20403', '1249', '1389', '1980', '1080', '12678', '22415', '12681', '3637', '12683', '12702', '12686', '41280', '991', '12685', '20004', '943', '1369', '20401', '22333', '3647', '21021', '12698', '23105', '12673', '23281', '42002', '20160', '20420', '22427', '12676', '971', '2443', '20432', '1289', '6177', '20161', '1070', '20404', '21003', '42012', '21000', '48', '20414', '12677', '20015', '23099', '22433', '42000', '6150', '42006', '864', '1269', '23101', '6153', '6162', '23102', '102', '1200', '94', '95', '23106', '1090', '20457', '2976', '12143', '31', '12680', '12682', '1259', '4056', '2624', '42004', '884', '20551', '20550', '12695', '20431', '20117', '894', '20416', '1558', '1618', '20456', '4080', '12340', '21002', '914', '23104', '3079', '1990', '41270', '12679', '42010', '2178']
cardiac_features.sort(key=int)
print('Number of cardiac features: ', len(cardiac_features))
masks = []
have_features = []
lack_features = []
for cf in cardiac_features:
    mask = data_df.columns.str.startswith(f'{cf}-')
    if not any(mask):
        lack_features.append(cf)
    else:
        have_features.append(cf)
    masks.append(mask)
print('Lack {} cardiac features: '.format(len(lack_features)), lack_features)
supermask = [any(l) for l in zip(*masks)]
supermask[0] = True  # keep eid
cardiac_cols = list(data_df.columns[supermask])

have_features_df = pd.DataFrame(
    {'FieldID': have_features,
     'FieldName': [field_id2name[int(i)] for i in have_features]}
)
have_features_df.to_csv('have_features.csv', index=False)

SAVE = False

if SAVE:
    chunk_size = 5000
    chunks = []
    i = 1
    for chunk in pd.read_csv(DATA_PATH, dtype=dtype, chunksize=chunk_size):
        chunks.append(chunk[cardiac_cols])
        print('Chunk {} done'.format(i))
        i += 1
    data_df = pd.concat(chunks)
    data_df = data_df[cardiac_cols]
    data_df.to_csv(CARDIAC_FEATURES_PATH, index=False)
    rename(data_df=data_df, datadict_df=datadict_df)
    data_df.to_csv(CARDIAC_PATIENTS_PATH, index=False)