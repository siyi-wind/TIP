import torch
import torch.nn as nn
from collections import OrderedDict

from models.TabularModel import TabularModel
from models.ImagingModel import ImagingModel
from pl_bolts.utils.self_supervised import torchvision_ssl_encoder

class MultimodalModelMUL(nn.Module):
  """
  Evaluation model for imaging and tabular data.
  For MUL method
  """
  def __init__(self, args) -> None:
    super(MultimodalModelMUL, self).__init__()

    self.args = args
    self.fusion_method = args.algorithm_name
    assert self.fusion_method == 'MUL'
    print('Fusion method: ', self.fusion_method)
    self.imaging_model = torchvision_ssl_encoder(args.model, return_all_feature_maps=True)
    self.tabular_layer1 = nn.Sequential(
                          nn.Linear(args.input_size, 64),
                          nn.ReLU(inplace=True), 
                          nn.BatchNorm1d(64)
                        )
    self.tabular_layer2 = nn.Sequential(
                      nn.Linear(64, 256),
                      nn.ReLU(inplace=True), 
                      nn.BatchNorm1d(256)
                    )
    self.tabular_layer3 = nn.Sequential(
                      nn.Linear(256, 512),
                      nn.ReLU(inplace=True), 
                      nn.BatchNorm1d(512)
                    )
    self.tabular_layer4 = nn.Sequential(
                      nn.Linear(512, 1024),
                      nn.ReLU(inplace=True), 
                      nn.BatchNorm1d(1024)
                    )
    self.tabular_layer5 = nn.Sequential(
                      nn.Linear(1024, 2048),
                      nn.ReLU(inplace=True), 
                    )
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
    elif isinstance(m, nn.BatchNorm1d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x_im = self.imaging_model.conv1(x[0])
    x_im = self.imaging_model.bn1(x_im)
    x_im = self.imaging_model.relu(x_im)
    x_im = self.imaging_model.maxpool(x_im)
    x_tab = self.tabular_layer1(x[1])
    x_im = x_tab.unsqueeze(2).unsqueeze(3) * x_im

    x_im = self.imaging_model.layer1(x_im)
    x_tab = self.tabular_layer2(x_tab)
    x_im = x_tab.unsqueeze(2).unsqueeze(3) * x_im

    x_im = self.imaging_model.layer2(x_im)
    x_tab = self.tabular_layer3(x_tab)
    x_im = x_tab.unsqueeze(2).unsqueeze(3) * x_im

    x_im = self.imaging_model.layer3(x_im)
    x_tab = self.tabular_layer4(x_tab)
    x_im = x_tab.unsqueeze(2).unsqueeze(3) * x_im

    x_im = self.imaging_model.layer4(x_im)
    x_tab = self.tabular_layer5(x_tab)
    x_im = x_tab.unsqueeze(2).unsqueeze(3) * x_im

    x = self.imaging_model.avgpool(x_im)
    x = torch.flatten(x, 1)

    x = self.head(x)
    return x