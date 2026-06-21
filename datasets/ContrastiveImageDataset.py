import random
from typing import List, Tuple

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_image
import numpy as np
import albumentations as A

def convert_to_float(x):
  return x.float()

def convert_to_ts(x, **kwargs):
  x = np.clip(x, 0, 255) / 255
  x = torch.from_numpy(x).float()
  x = x.permute(2,0,1)
  return x

def convert_to_ts_01(x, **kwargs):
  x = torch.from_numpy(x).float()
  x = x.permute(2,0,1)
  return x

class ContrastiveImageDataset(Dataset):
  """
  Dataset of images that serves two views of a subjects image and their label.
  Can delete first channel (segmentation channel) if specified
  """
  def __init__(self, data: str, labels: str, transform: transforms.Compose, delete_segmentation: bool, augmentation_rate: float, img_size: int, live_loading: bool,
               target: str, augmentation_speedup: bool=False) -> None:
    """
    data:                 Path to torch file containing images
    labels:               Path to torch file containing labels
    transform:            Compiled torchvision augmentations
    delete_segmentation:  If true, removes first channel from all images
    sim_matrix_path:      Path to file containing similarity matrix of subjects
    target:               DVM/CAD/Infarction
    """
    self.data = torch.load(data)
    self.labels = torch.load(labels)
    self.transform = transform
    self.augmentation_rate = augmentation_rate
    self.augmentation_speedup = augmentation_speedup
    self.live_loading = live_loading
    self.target = target
    if delete_segmentation:
      for im in self.data:
        im[0,:,:] = 0

    if augmentation_speedup:
      if self.target == 'dvm':
        self.default_transform = A.Compose([
          A.Resize(height=img_size, width=img_size),
          A.Lambda(name='convert2tensor', image=convert_to_ts)
        ])
        print(f'Using dvm transform for default transform in ContrastiveImageDataset')
      elif self.target == 'Infarction' or self.target == 'CAD':
        self.default_transform = A.Compose([
          A.Resize(height=img_size, width=img_size),
          A.Lambda(name='convert2tensor', image=convert_to_ts_01)
        ])
        print(f'Using cardiac transform for default transform in ContrastiveImageDataset')
      else:
        raise print('Only support dvm and cardiac datasets')
    else:
      self.default_transform = transforms.Compose([
        transforms.Resize(size=(img_size,img_size)),
        # transforms.Lambda(lambda x : x.float())
        transforms.Lambda(convert_to_float)
      ])


    # self.default_transform = transforms.Compose([
    #   transforms.Resize(size=(img_size,img_size)),
    #   transforms.Lambda(lambda x : x.float())
    # ])

  def __getitem__(self, indx: int) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """
    Returns two augmented views of one image and its label
    """
    view_1, view_2 = self.generate_imaging_views(indx)

    return view_1, view_2, self.labels[indx]

  def __len__(self) -> int:
    return len(self.data)

  def generate_imaging_views(self, index: int) -> List[torch.Tensor]:
    """
    Generates two views of a subjects image. 
    The first is always augmented. The second has {augmentation_rate} chance to be augmented.
    """
    im = self.data[index]
    if self.live_loading:
        if self.augmentation_speedup:
          im = np.load(im[:-4]+'.npy', allow_pickle=True)
        else:
          im = read_image(im)
          im = im / 255 if self.dataset_name == 'dvm' else im

    view_1 = self.transform(image=im)['image'] if self.augmentation_speedup else self.transform(im)
    if random.random() < self.augmentation_rate:
      view_2 = self.transform(image=im)['image'] if self.augmentation_speedup else self.transform(im)
    else:
      view_2 = self.default_transform(image=im)['image'] if self.augmentation_speedup else self.default_transform(im)
    
    return view_1, view_2