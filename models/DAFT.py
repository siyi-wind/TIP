'''
Based on DAFT codebase https://github.com/ai-med/DAFT
'''
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import math
from pl_bolts.utils.self_supervised import torchvision_ssl_encoder


class DAFT_block(nn.Module):
  def __init__(self, image_dim, tabular_dim, r=7) -> None:
    super(DAFT_block, self).__init__()
    self.global_pool = nn.AdaptiveAvgPool2d((1,1))
    h1 = image_dim+tabular_dim
    h2 = int(h1/r)
    self.multimodal_projection = nn.Sequential(
                              nn.Linear(h1, h2),
                              nn.ReLU(inplace=True),
                              nn.Linear(h2, 2*image_dim)
                              )
  
  def forward(self, x_im, x_tab):
    B,C,H,W = x_im.shape
    x = self.global_pool(x_im).squeeze()
    x = torch.cat([x, x_tab], dim=1)
    attention = self.multimodal_projection(x)

    v_scale, v_shift = torch.split(attention, C, dim=1)
    v_scale = v_scale.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,H,W)
    v_shift = v_shift.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,H,W)
    x = v_scale * x_im + v_shift
    return x


class DAFT(nn.Module):
  """
  Evaluation model for imaging and tabular data.
  """
  def __init__(self, args) -> None:
    super(DAFT, self).__init__()
    self.args = args
    self.imaging_encoder = torchvision_ssl_encoder(args.model, return_all_feature_maps=True)
    self.imaging_encoder.layer4 = torch.nn.Sequential(*list(self.imaging_encoder.layer4.children())[:-1])
    self.tabular_encoder = nn.Identity()
    self.daft = DAFT_block(args.embedding_dim, args.input_size)

    in_ch, out_ch = args.embedding_dim//4, args.embedding_dim
    self.residual = nn.Sequential(nn.Conv2d(out_ch, in_ch, kernel_size=1, bias=False),
                                  nn.BatchNorm2d(in_ch),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=False),
                                  nn.BatchNorm2d(in_ch),
                                  nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                                  nn.BatchNorm2d(out_ch)
                                            )
    self.shortcut = nn.Identity()

    self.act = nn.ReLU(inplace=True)
    in_dim = args.embedding_dim
    self.head = nn.Linear(in_dim, args.num_classes)

    self.apply(self.init_weights)

  def init_weights(self, m: nn.Module, init_gain = 0.02) -> None:
    """
    Initializes weights according to desired strategy
    """
    if isinstance(m, nn.Linear):
      if self.args.init_strat == 'normal':
        nn.init.normal_(m.weight.data, 0, 0.001)
      elif self.args.init_strat == 'xavier':
        nn.init.xavier_normal_(m.weight.data, gain=init_gain)
      elif self.args.init_strat == 'kaiming':
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
      elif self.args.init_strat == 'orthogonal':
        nn.init.orthogonal_(m.weight.data, gain=init_gain)
      if hasattr(m, 'bias') and m.bias is not None:
        nn.init.constant_(m.bias.data, 0.0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x_im = self.imaging_encoder(x[0])[-1]
    x = self.daft(x_im=x_im, x_tab=x[1])
    x = self.residual(x)
    x = x + self.shortcut(x_im)
    x = self.act(x)
    x = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
    x = self.head(x)
    return x


if __name__ == "__main__":
  model = DAFT_block(128, 90)
  img = torch.randn(4,128,7,7)
  tab = torch.randn(4,90)
  y = model(img, tab)
  print(y.shape)