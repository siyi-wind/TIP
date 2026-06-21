from typing import List, Tuple
import random
import csv
import copy

import torch
from torch.utils.data import Dataset
import pandas as pd
from torchvision.transforms import transforms
from torchvision.io import read_image
import albumentations as A
import numpy as np


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


class ContrastiveImagingAndTabularDataset(Dataset):
  """
  Multimodal dataset that generates multiple views of imaging and tabular data for contrastive learning.

  The first imaging view is always augmented. The second has {augmentation_rate} chance of being augmented.
  The first tabular view is never augmented. The second view is corrupted by replacing {corruption_rate} features
  with values chosen from the empirical marginal distribution of that feature.
  """
  def __init__(
      self, 
      data_path_imaging: str, delete_segmentation: bool, augmentation: transforms.Compose, augmentation_rate: float, 
      data_path_tabular: str, corruption_rate: float, field_lengths_tabular: str, one_hot_tabular: bool,
      labels_path: str, img_size: int, live_loading: bool, augmentation_speedup: bool=False) -> None:
            
    # Imaging
    self.data_imaging = torch.load(data_path_imaging)
    self.transform = augmentation
    self.delete_segmentation = delete_segmentation
    self.augmentation_rate = augmentation_rate
    self.live_loading = live_loading
    self.augmentation_speedup = augmentation_speedup
    self.dataset_name = data_path_tabular.split('/')[-1].split('_')[0]

    if self.delete_segmentation:
      for im in self.data_imaging:
        im[0,:,:] = 0

    if augmentation_speedup:
      if self.dataset_name == 'dvm':
        self.default_transform = A.Compose([
          A.Resize(height=img_size, width=img_size),
          A.Lambda(name='convert2tensor', image=convert_to_ts)
        ])
        print(f'Using dvm transform for default transform in ContrastiveImagingAndTabularDataset')
      elif self.dataset_name == 'cardiac':
        self.default_transform = A.Compose([
          A.Resize(height=img_size, width=img_size),
          A.Lambda(name='convert2tensor', image=convert_to_ts_01)
        ])
        print(f'Using cardiac transform for default transform in ContrastiveImagingAndTabularDataset')
      else:
        raise print('Only support dvm and cardiac datasets')
    else:
      self.default_transform = transforms.Compose([
        transforms.Resize(size=(img_size,img_size)),
        # transforms.Lambda(lambda x : x.float())
        transforms.Lambda(convert_to_float)
      ])

    # Tabular
    self.data_tabular = self.read_and_parse_csv(data_path_tabular)
    self.generate_marginal_distributions(data_path_tabular)
    self.c = corruption_rate
    self.field_lengths_tabular = torch.load(field_lengths_tabular)
    self.one_hot_tabular = one_hot_tabular

    # Change the order of features to categorical, continuous 
    
    # Classifier
    self.labels = torch.load(labels_path)
  
  def read_and_parse_csv(self, path_tabular: str) -> List[List[float]]:
    """
    Does what it says on the box.
    """
    with open(path_tabular,'r') as f:
      reader = csv.reader(f)
      data = []
      for r in reader:
        r2 = [float(r1) for r1 in r]
        data.append(r2)
    return data

  def generate_marginal_distributions(self, data_path) -> None:
    """
    Generates empirical marginal distribution by transposing data
    """
    # data = np.array(self.data_tabular)
    # self.marginal_distributions = np.transpose(data)
    data_df = pd.read_csv(data_path, header=None)
    self.marginal_distributions = data_df.transpose().values.tolist()

  def get_input_size(self) -> int:
    """
    Returns the number of fields in the table. 
    Used to set the input number of nodes in the MLP
    """
    if self.one_hot_tabular:
      return int(sum(self.field_lengths_tabular))
    else:
      return len(self.data[0])

  def corrupt(self, subject: List[float]) -> List[float]:
    """
    Creates a copy of a subject, selects the indices 
    to be corrupted (determined by hyperparam corruption_rate)
    and replaces their values with ones sampled from marginal distribution
    """
    subject = copy.deepcopy(subject)

    indices = random.sample(list(range(len(subject))), int(len(subject)*self.c)) 
    for i in indices:
      subject[i] = random.sample(self.marginal_distributions[i],k=1)[0] 
    return subject

  def one_hot_encode(self, subject: torch.Tensor) -> torch.Tensor:
    """
    One-hot encodes a subject's features
    """
    out = []
    for i in range(len(subject)):
      if self.field_lengths_tabular[i] == 1:
        out.append(subject[i].unsqueeze(0))
      else:
        out.append(torch.nn.functional.one_hot(subject[i].long(), num_classes=int(self.field_lengths_tabular[i])))
    return torch.cat(out)

  def generate_imaging_views(self, index: int) -> List[torch.Tensor]:
    """
    Generates two views of a subjects image. Also returns original image resized to required dimensions.
    The first is always augmented. The second has {augmentation_rate} chance to be augmented.
    """
    im = self.data_imaging[index]
    if self.live_loading:
      if self.augmentation_speedup:
        im = np.load(im[:-4]+'.npy', allow_pickle=True)
      else:
        im = read_image(im)
        im = im / 255 if self.dataset_name == 'dvm' else im
    # ims = [self.transform(image=im)['image']] if self.augmentation_speedup else [self.transform(im)]
    ims = [torch.tensor(0, dtype=torch.float)] # Placeholder
    if random.random() < self.augmentation_rate:
      ims.append(self.transform(image=im)['image'] if self.augmentation_speedup else self.transform(im))
    else:
      ims.append(self.default_transform(image=im)['image'] if self.augmentation_speedup else self.default_transform(im))

    orig_im = self.default_transform(image=im)['image'] if self.augmentation_speedup else self.default_transform(im)
    
    return ims, orig_im

  def __getitem__(self, index: int) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor]:
    imaging_views, unaugmented_image = self.generate_imaging_views(index)
    tabular_views = [torch.tensor(self.data_tabular[index], dtype=torch.float), torch.tensor(self.corrupt(self.data_tabular[index]), dtype=torch.float)]
    if self.one_hot_tabular:
      tabular_views = [self.one_hot_encode(tv) for tv in tabular_views]
    label = torch.tensor(self.labels[index], dtype=torch.long)
    return imaging_views, tabular_views, label, unaugmented_image

  def __len__(self) -> int:
    return len(self.data_tabular)