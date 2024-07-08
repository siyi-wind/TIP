'''
* Licensed under the Apache License, Version 2.
* By Siyi Du, 2024
'''
import os
import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt


# convert jpg image to numpy array
def process_DVM(
    DVM_feature_folder = '/home/siyi/data/DVM/features'): 
    for split in ['train', 'val', 'test']:
        img_paths  = torch.load(os.path.join(DVM_feature_folder, '{}_paths_all_views.pt'.format(split)))
        np_paths = []
        for path in tqdm(img_paths):
            img_np = plt.imread(path)
            save_path = path[:-4] + '.npy'
            np.save(save_path, img_np)
            np_paths.append(save_path)
        #     break
        # break
    return

            

if __name__ == '__main__':
    process_DVM('/home/siyi/data/DVM/features')