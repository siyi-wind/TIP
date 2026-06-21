import random
import csv
import copy
from typing import List, Tuple

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class MaskTabularDataset(Dataset):
  """
  Follow VIME https://github.com/jsyoon0823/VIME/tree/master
  Dataset of tabular data that generates two views and a mask.
  one untouched and one corrupted and mask pos==1 if this pos is unequal
  The corrupted view hsd a random fraction is replaced with values sampled 
  from the empirical marginal distribution of that value
  """
  def __init__(self, data_path: str, labels_path: str, corruption_rate: float=0.6, field_lengths_tabular: str=None, one_hot: bool=True):
    self.data = np.array(self.read_and_parse_csv(data_path))
    self.labels = torch.load(labels_path)
    self.c = corruption_rate
    # self.generate_marginal_distributions()

    self.field_lengths = torch.load(field_lengths_tabular)

    self.one_hot = one_hot
    m_unlab = self.mask_generator(corruption_rate, self.data)
    self.mask_labels, self.data_corrupted = self.pretext_generator(m_unlab, self.data)

    assert self.data.shape[0] == len(self.labels) == self.mask_labels.shape[0] == self.data_corrupted.shape[0]
  
  def mask_generator(self, p_m, x):
    mask = np.random.binomial(1, p_m, x.shape)
    return mask
  
  def pretext_generator (self, m, x):  
    """Generate corrupted samples.
    Args:
      m: mask matrix
      x: feature matrix
      
    Returns:
      m_new: final mask matrix after corruption
      x_tilde: corrupted feature matrix
    """
    # Parameters
    no, dim = x.shape  
    # Randomly (and column-wise) shuffle data
    x_bar = np.zeros([no, dim])
    for i in range(dim):
      idx = np.random.permutation(no)
      x_bar[:, i] = x[idx, i]
    
    # Corrupt samples
    x_tilde = x * (1-m) + x_bar * m  
    # Define new mask matrix
    m_new = 1 * (x != x_tilde)
    return m_new, x_tilde
  
  def read_and_parse_csv(self, path: str) -> List[List[float]]:
    """
    Does what it says on the box.
    """
    with open(path,'r') as f:
      reader = csv.reader(f)
      data = []
      for r in reader:
        r2 = [float(r1) for r1 in r]
        data.append(r2)
    return data

  def generate_marginal_distributions(self) -> None:
    """
    Generates empirical marginal distribution by transposing data
    """
    data = np.array(self.data)
    self.marginal_distributions = np.transpose(data)
    # data_df = pd.read_csv(data_path)
    # self.marginal_distributions = data_df.transpose().values.tolist()

  def get_input_size(self) -> int:
    """
    Returns the number of fields in the table. 
    Used to set the input number of nodes in the MLP
    """
    if self.one_hot:
      return int(sum(self.field_lengths))
    else:
      return len(self.data[0])

  def corrupt(self, subject: List[float]) -> List[float]:
    """
    Creates a copy of a subject, selects the indices 
    to be corrupted (determined by hyperparam corruption_rate)
    and replaces their values with ones sampled from marginal distribution
    """
    subject = copy.deepcopy(subject)
    subject = np.array(subject)

    indices = random.sample(list(range(len(subject))), int(len(subject)*self.c)) 
    pick_value_positions = np.random.choice(self.marginal_distributions.shape[1], size=len(indices))
    subject[indices] = self.marginal_distributions[indices, pick_value_positions]
    return subject
  
    # subject = copy.deepcopy(subject)

    # indices = random.sample(list(range(len(subject))), int(len(subject)*self.c)) 
    # for i in indices:
    #   subject[i] = random.sample(self.marginal_distributions[i],k=1)[0] 
    # return subject

  def one_hot_encode(self, subject: torch.Tensor) -> torch.Tensor:
    """
    One-hot encodes a subject's features
    """
    out = []
    for i in range(len(subject)):
      if self.field_lengths[i] == 1:
        out.append(subject[i].unsqueeze(0))
      else:
        out.append(torch.nn.functional.one_hot(subject[i].long(), num_classes=int(self.field_lengths[i])))
    return torch.cat(out)

  def __getitem__(self, index: int) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """
    Returns two views of a subjects features, the first element being the original subject features
    and the second element being the corrupted view. Also returns the label of the subject
    """
    corrupted_item = torch.tensor(self.data_corrupted[index], dtype=torch.float)
    uncorrupted_item = torch.tensor(self.data[index], dtype=torch.float)
    mask = torch.tensor(self.mask_labels[index], dtype=torch.float)
    if self.one_hot:
      corrupted_item = self.one_hot_encode(corrupted_item)
      # uncorrupted_item = self.one_hot_encode(uncorrupted_item)
    
    item = (uncorrupted_item, mask), corrupted_item, torch.tensor(self.labels[index], dtype=torch.long)
    return item


    # corrupted_item = torch.tensor(self.corrupt(self.data[index]), dtype=torch.float)
    # uncorrupted_item = torch.tensor(self.data[index], dtype=torch.float)
    # if self.one_hot:
    #   corrupted_item = self.one_hot_encode(corrupted_item)
    #   uncorrupted_item = self.one_hot_encode(uncorrupted_item)
    # item = uncorrupted_item, corrupted_item, torch.tensor(self.labels[index], dtype=torch.long)
    # return item

  def __len__(self) -> int:
    return len(self.labels)
