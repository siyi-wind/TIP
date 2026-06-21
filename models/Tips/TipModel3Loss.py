'''
* Licensed under the Apache License, Version 2.
* By Siyi Du, 2024
'''

from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.clip_loss import CLIPLoss
from utils.reconstruct_loss import ReconstructionLoss

from models.Tip_utils.Tip_pretraining import Pretraining


class TIP3Loss(Pretraining):
    '''
    Tabular-Imaging Pretraining
    '''
    def __init__(self, hparams) -> None:
        super().__init__(hparams)

        # Imaging
        self.initialize_imaging_encoder_and_projector()
        
        if self.hparams.imaging_pretrain_checkpoint:
            self.load_pretrained_imaging_weights()
        
        # Tabular 
        self.initialize_tabular_encoder_and_projector()

        # Multimodal
        self.initialize_multimodal_encoder_and_predictor()

        # image tabular matching 
        self.itm_head = nn.Linear(self.hparams.multimodal_embedding_dim, 2)

        # loss
        nclasses = hparams.batch_size
        self.criterion_val_itc = CLIPLoss(temperature=self.hparams.temperature, lambda_0=self.hparams.lambda_0)
        self.criterion_train_itc = self.criterion_val_itc
        self.criterion_tr = ReconstructionLoss(num_cat=self.hparams.num_cat, cat_offsets=self.encoder_tabular.cat_offsets, num_con=self.hparams.num_con)
        self.criterion_itm = nn.CrossEntropyLoss(reduction='mean')
        
        self.initialize_classifier_and_metrics(nclasses, nclasses)

        print(f'Tabular model, multimodal: {self.encoder_tabular}\n{self.projector_tabular}')
        print(f'Imaging model, multimodal: {self.encoder_imaging}\n{self.projector_imaging}')
        print(f'Multimodal model: {self.encoder_multimodal}')
        print(f'Predictor model, tabular: {self.predictor_tabular}')
    
    def cal_image_tabular_matching_loss(self, image_embeddings: torch.Tensor, tabular_embeddings: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        current_device = image_embeddings.device
        output_pos = self.forward_multimodal_feature(tabular_features=tabular_embeddings, image_features=image_embeddings)
        B = image_embeddings.shape[0]
        # get negative pairs
        with torch.no_grad():
            weights_i2t = F.softmax(logits, dim=1)+1e-4
            weights_i2t.fill_diagonal_(0)
            weights_t2i = F.softmax(logits.T, dim=1)+1e-4
            weights_t2i.fill_diagonal_(0)
        
        tabular_embeddings_neg = torch.zeros_like(tabular_embeddings).to(current_device)
        for b in range(B):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            tabular_embeddings_neg[b] = tabular_embeddings[neg_idx]

        image_embeddings_neg = torch.zeros_like(image_embeddings).to(current_device)
        for b in range(B):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            image_embeddings_neg[b] = image_embeddings[neg_idx]

        tabular_embeddings_all = torch.cat([tabular_embeddings, tabular_embeddings_neg], dim=0)
        image_embeddings_all = torch.cat([image_embeddings_neg, image_embeddings], dim=0)
        output_neg = self.forward_multimodal_feature(tabular_features=tabular_embeddings_all, image_features=image_embeddings_all)
        z = self.itm_head(torch.cat([output_pos, output_neg], dim=0))
        itm_labels = torch.cat([torch.ones(B), torch.zeros(2*B)], dim=0).long().to(logits.device)
        loss_itm = self.criterion_itm(z, itm_labels)
        return loss_itm, z, itm_labels


    def training_step(self, batch: Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor, List[torch.Tensor]], _) -> torch.Tensor:
        '''
        Train
        Tabular-imaging contrastive learning
        Tabular reconstruction learning
        '''
        im_views, tab_views, y, _, original_tab = batch

        # =======================================  itc    =======================================================================
        # Augmented image and unagumented tabular
        z0, image_embeddings = self.forward_imaging(im_views[1]) 
        z1, tabular_embeddings = self.forward_tabular(tab_views[0])
        loss_itc, logits, labels = self.criterion_train_itc(z0, z1, y)
        self.log(f"multimodal.train.ITCloss", loss_itc, on_epoch=True, on_step=False)

        # =======================================  itm  =======================================================================
        loss_itm, logits_itm, labels_itm = self.cal_image_tabular_matching_loss(image_embeddings, tabular_embeddings, logits)
        self.log(f"multimodal.train.ITMloss", loss_itm, on_epoch=True, on_step=False)

        # =======================================  tr    =======================================================================
        # masked tabular 
        mask, mask_special = tab_views[2], tab_views[3]
        _, tabular_embeddings = self.forward_tabular(tab_views[1], mask=mask, mask_special=mask_special)
        z2, multimodal_embeddings = self.forward_multimodal(tabular_features=tabular_embeddings, image_features=image_embeddings)
        loss_tr, pred_cat, target_cat, mask_cat = self.criterion_tr(z2,original_tab,mask=mask)
        self.log(f"multimodal.train.TRloss", loss_tr, on_epoch=True, on_step=False)
        
        if len(im_views[0])==self.hparams.batch_size:
            self.calc_and_log_train_embedding_acc(logits=logits, labels=labels, modality='multimodal')
            self.calc_and_log_train_cat_embedding_acc(logits=pred_cat, labels=target_cat, mask=mask_cat, modality='multimodal')
            self.calc_and_log_train_itm_acc(logits=logits_itm, labels=labels_itm, modality='multimodal')
        
        loss = (loss_itc + loss_tr + loss_itm)/3.0
        # loss = (loss_itc + loss_tr)/3.0
        self.log(f"multimodal.train.loss", loss, on_epoch=True, on_step=False)
        return {'loss':loss, 'embeddings': multimodal_embeddings, 'labels': y, }

    
    def validation_step(self, batch: Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor, List[torch.Tensor]], _) -> torch.Tensor:
        '''
        Validate
        Tabular-imaging contrastive learning
        Tabular reconstruction learning
        '''
        im_views, tab_views, y, original_im, original_tab = batch

        # =======================================  itc    =======================================================================
        # Unaugmented views
        z0, image_embeddings = self.forward_imaging(original_im) 
        z1, tabular_embeddings = self.forward_tabular(original_tab)
        loss_itc, logits, labels = self.criterion_val_itc(z0, z1, y)
        self.log(f"multimodal.val.ITCloss", loss_itc, on_epoch=True, on_step=False)

        # =======================================  itm  =======================================================================
        loss_itm, logits_itm, labels_itm = self.cal_image_tabular_matching_loss(image_embeddings, tabular_embeddings, logits)
        self.log(f"multimodal.val.ITMloss", loss_itm, on_epoch=True, on_step=False)

        # =======================================  tr    =======================================================================
        # masked tabular 
        mask, mask_special = tab_views[2], tab_views[3]
        _, tabular_embeddings = self.forward_tabular(tab_views[1], mask=mask, mask_special=mask_special)
        z2, multimodal_embeddings = self.forward_multimodal(tabular_features=tabular_embeddings, image_features=image_embeddings)
        loss_tr, pred_cat, target_cat, mask_cat = self.criterion_tr(z2,original_tab, mask=mask)
        self.log(f"multimodal.val.TRloss", loss_tr, on_epoch=True, on_step=False)

        if len(im_views[0])==self.hparams.batch_size:
            self.calc_and_log_val_embedding_acc(logits=logits, labels=labels, modality='multimodal')
            self.calc_and_log_val_cat_embedding_acc(logits=pred_cat, labels=target_cat, mask=mask_cat, modality='multimodal')
            self.calc_and_log_val_itm_acc(logits=logits_itm, labels=labels_itm, modality='multimodal')
        
        loss = (loss_itc + loss_tr + loss_itm)/3.0
        # loss = (loss_itc + loss_tr)/3.0
        self.log(f"multimodal.val.loss", loss, on_epoch=True, on_step=False)
        return {'sample_augmentation': im_views[1], 'embeddings': multimodal_embeddings, 'labels': y}
    

    def configure_optimizers(self) -> Tuple[Dict, Dict]:
        """
        Define and return optimizer and scheduler for contrastive model. 
        """
        optimizer = torch.optim.Adam(
        [
            {'params': self.encoder_imaging.parameters()}, 
            {'params': self.projector_imaging.parameters()},
            {'params': self.encoder_tabular.parameters()},
            {'params': self.projector_tabular.parameters()},
            {'params': self.encoder_multimodal.parameters()},
            {'params': self.predictor_tabular.parameters()},
            {'params': self.itm_head.parameters()},
        ], lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        
        scheduler = self.initialize_scheduler(optimizer)
        
        return (
        { # Contrastive
            "optimizer": optimizer, 
            "lr_scheduler": scheduler
        }
        )
    
