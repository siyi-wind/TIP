import torch
import torch.nn as nn
from collections import OrderedDict

from models.TabularModel import TabularModel
from models.ImagingModel import ImagingModel

class MultimodalModel(nn.Module):
  """
  Evaluation model for imaging and tabular data.
  For MAX and CONCAT methods
  """
  def __init__(self, args) -> None:
    super(MultimodalModel, self).__init__()

    self.imaging_model = ImagingModel(args)
    self.tabular_model = TabularModel(args)
    self.fusion_method = args.algorithm_name
    print('Fusion method: ', self.fusion_method)
    if self.fusion_method in set(['MAX']):
      self.imaging_proj = nn.Linear(args.embedding_dim, args.tabular_embedding_dim)
      in_dim = args.tabular_embedding_dim
    elif self.fusion_method == 'CONCAT':
      in_dim = args.embedding_dim + args.tabular_embedding_dim
    else:
      raise ValueError('Fusion method not recognized.')
    self.head = nn.Linear(in_dim, args.num_classes)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x_im = self.imaging_model.encoder(x[0])[0].squeeze()
    x_tab = self.tabular_model.encoder(x[1]).squeeze()

    if self.fusion_method == 'CONCAT':
      x = torch.cat([x_im, x_tab], dim=1)
    elif self.fusion_method == 'MAX':
      x_im = self.imaging_proj(x_im)
      x = torch.stack([x_im, x_tab], dim=1)
      x, _ = torch.max(x, dim=1)

    x = self.head(x)
    return x