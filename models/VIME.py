'''
Based on VIME codebase  https://github.com/jsyoon0823/VIME
'''
from typing import List, Tuple, Dict, Any

import torch
import torch.nn as nn

from utils.ntx_ent_loss_custom import NTXentLoss
from models.pretraining import Pretraining


class VIME(Pretraining):
  """
  Lightning module for VIME pretraining. 
  """
  def __init__(self, hparams) -> None:
    super().__init__(hparams)

    self.initialize_tabular_encoder()

    # balance between mask and reconstruction loss
    self.alpha = 2.0
    self.num_cat = len(self.cat_lengths_tabular)
    self.num_con = len(self.con_lengths_tabular)
    self.num_unique_cat = sum(self.cat_lengths_tabular)
    self.mask_classifier = nn.Linear(hparams.tabular_embedding_dim, self.num_cat+self.num_con)
    self.cat_classifier = nn.Linear(hparams.tabular_embedding_dim, self.num_unique_cat*self.num_cat)
    self.con_regressor = nn.Linear(hparams.tabular_embedding_dim, self.num_con)

    self.pooled_dim = self.hparams.tabular_embedding_dim

    self.criterion_mask = nn.BCEWithLogitsLoss()
    self.criterion_con = nn.MSELoss()
    self.criterion_cat = nn.CrossEntropyLoss()

    # for contrastive learning didn't use
    nclasses = hparams.batch_size
    self.initialize_classifier_and_metrics(nclasses, nclasses)

    print(self.encoder_tabular)
    print(self.mask_classifier)
    print(self.cat_classifier)
    print(self.con_regressor)
  
  
  def forward(self, x: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Forward pass for tabular
    """
    embeddings = self.encoder_tabular(x)
    return embeddings

  def training_step(self, batch: Tuple[List[torch.Tensor], torch.Tensor], _) -> torch.Tensor:
    """
    Trains contrastive model
    """
    views, x, y = batch 
    B, N = x.shape
    # original_tab, mask = views
    embeddings = self.encoder_tabular(x)
    # mask classification
    z_m = self.mask_classifier(embeddings)
    loss_m = self.criterion_mask(input=z_m, target=views[1])
    # cat classification
    z_cat = self.cat_classifier(embeddings)
    z_cat = z_cat.reshape(B*self.num_cat, self.num_unique_cat)
    target_cat = views[0][:,:self.num_cat].reshape(B*self.num_cat).long()
    loss_cat = self.criterion_cat(input=z_cat, target=target_cat)
    # con regression
    z_con = self.con_regressor(embeddings)
    target_con = views[0][:,self.num_cat:]
    loss_con = self.criterion_con(input=z_con, target=target_con)

    loss = loss_m + (loss_cat+loss_con)/2.0*self.alpha

    self.log('tabular.train.TRloss', (loss_cat+loss_con)/2, on_epoch=True, on_step=False)
    self.log('tabular.train.loss', loss, on_epoch=True, on_step=False)

    return {'loss':loss, 'embeddings': embeddings, 'labels': y}


  def validation_step(self, batch: Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor], _) -> None:
    """
    Validate both contrastive model and classifier
    """
    views, x, y = batch 
    B, N = x.shape
    # original_tab, mask = views
    embeddings = self.encoder_tabular(x)
    # mask classification
    z_m = self.mask_classifier(embeddings)
    loss_m = self.criterion_mask(input=z_m, target=views[1])
    # cat classification
    z_cat = self.cat_classifier(embeddings)
    z_cat = z_cat.reshape(B*self.num_cat, self.num_unique_cat)
    target_cat = views[0][:,:self.num_cat].reshape(B*self.num_cat).long()
    loss_cat = self.criterion_cat(input=z_cat, target=target_cat)
    # con regression
    z_con = self.con_regressor(embeddings)
    target_con = views[0][:,self.num_cat:]
    loss_con = self.criterion_con(input=z_con, target=target_con)

    loss = loss_m + (loss_cat+loss_con)/2.0*self.alpha

    self.log('tabular.val.TRloss', (loss_cat+loss_con)/2, on_epoch=True, on_step=False)
    self.log('tabular.val.loss', loss, on_epoch=True, on_step=False)

    return {'loss':loss, 'embeddings': embeddings, 'labels': y}

  
  def configure_optimizers(self) -> Tuple[Dict, Dict]:
    """
    Define and return optimizer and scheduler for contrastive model and online classifier. 
    Scheduler for online classifier often disabled
    """
    optimizer = torch.optim.Adam(
      [
        {'params': self.encoder_tabular.parameters()}, 
        {'params': self.mask_classifier.parameters()},
        {'params': self.cat_classifier.parameters()},
        {'params': self.con_regressor.parameters()}
      ], lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
    
    scheduler = self.initialize_scheduler(optimizer)
    
    return (
      { # Contrastive
        "optimizer": optimizer, 
        "lr_scheduler": scheduler
      }
    )