'''
* Licensed under the Apache License, Version 2.
* By Siyi Du, 2024
* Based on Vision Transformer and BERT
* Based on AViT https://github.com/siyi-wind/AViT/blob/main/Models/Transformer/ViT_adapters.py
* Based on BLIP https://github.com/salesforce/BLIP/blob/main/models/med.py
'''
from typing import Dict, List
from collections import OrderedDict
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange,repeat
import sys
# TODO: Change the path to your own project directory if you want to run this file alone for debugging 
sys.path.append('/home/siyi/project/mm/multimodal/TIP')
from models.Tip_utils.pieces import DotDict

import torch
import torch.nn as nn


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., with_qkv=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.with_qkv = with_qkv
        if self.with_qkv:
           self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
           self.proj = nn.Linear(dim, dim)
           self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.save_attention = False
        self.save_gradients = False

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients
        
    def get_attn_gradients(self):
        return self.attn_gradients
    
    def save_attention_map(self, attention_map):
        self.attention_map = attention_map
        
    def get_attention_map(self):
        return self.attention_map

    def forward(self, x, mask=None, visualize=False):
        B, N, C = x.shape
        if self.with_qkv:
           qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
           q, k, v = qkv[0], qkv[1], qkv[2]
        else:
           qkv = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
           q, k, v  = qkv, qkv, qkv

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            attn = attn + mask

        attn = attn.softmax(dim=-1)
        if self.save_attention:
            self.save_attention_map(attn)
        if self.save_gradients:
            attn.register_hook(self.save_attn_gradients)
        attn = self.attn_drop(attn)
        # print(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        if self.with_qkv:
           x = self.proj(x)
           x = self.proj_drop(x)
        if visualize == False:
            return x
        else:
            return x, attn


class CrossAttention(nn.Module):
    def __init__(self, q_dim, k_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., with_qkv=True):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        head_dim = k_dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.with_qkv = with_qkv
        self.kv_proj = nn.Linear(k_dim,k_dim*2,bias=qkv_bias)
        self.q_proj = nn.Linear(q_dim,k_dim)
        self.proj = nn.Linear(k_dim, k_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.save_attention = False
        self.save_gradients = False

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients
        
    def get_attn_gradients(self):
        return self.attn_gradients
    
    def save_attention_map(self, attention_map):
        self.attention_map = attention_map
        
    def get_attention_map(self):
        return self.attention_map

    def forward(self, q, k, visualize=False):
        B,N_k,K = k.shape
        _,N_q,_ = q.shape
        kv = self.kv_proj(k).reshape(B,N_k,2,self.num_heads,K//self.num_heads).permute(2, 0, 3, 1, 4)  # 
        k,v = kv[0], kv[1]  # (B,H,N,C)
        q = self.q_proj(q).reshape(B,N_q,self.num_heads,K//self.num_heads).permute(0,2,1,3)  # (B,H,N,C)
        attn = (q @ k.transpose(-2,-1))*self.scale
        attn = attn.softmax(dim=-1)
        if self.save_attention:
            self.save_attention_map(attn)
        if self.save_gradients:
            attn.register_hook(self.save_attn_gradients)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N_q, K)
        out = self.proj(out)
        out = self.proj_drop(out)
        if visualize == False:
            return out
        else:
            return out, attn


class Block(nn.Module):
    def __init__(self, dim, num_heads=8, is_cross_attention=False, encoder_dim=None, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.scale = 0.5
        self.norm1 = norm_layer(dim)
        self.is_cross_attention = is_cross_attention
        self.attn = Attention(
        dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        if self.is_cross_attention:
           self.cross_attn = CrossAttention(
               q_dim=dim, k_dim=encoder_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
           self.cross_norm = norm_layer(dim)

        ## drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, encoder_hidden_states=None, mask=None, visualize=False):
        if visualize==False:
            # self attention
            x = x + self.drop_path(self.attn(self.norm1(x), mask=mask))
            # cross attention
            if self.is_cross_attention:
                assert encoder_hidden_states is not None
                x = x + self.drop_path(self.cross_attn(self.cross_norm(x), encoder_hidden_states))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x
        else:
            tmp, self_attn = self.attn(self.norm1(x), mask=mask, visualize=visualize)
            x = x+self.drop_path(tmp)
            if self.is_cross_attention:
                assert encoder_hidden_states is not None      
                tmp, cross_attn = self.cross_attn(self.cross_norm(x), encoder_hidden_states, visualize=visualize)
                x = x+self.drop_path(tmp)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x, {'self_attn':self_attn, 'cross_attn':cross_attn if self.is_cross_attention else None}


class TabularTransformerEncoder(nn.Module):
    '''
    Tabular Transformer Encoder based on BERT
    cat_lengths_tabular: categorical feature length list, e.g., [5,4,2]
    con_lengths_tabular: continuous feature length list, e.g., [1,1]
    '''

    def __init__(self, args: Dict, cat_lengths_tabular: List, con_lengths_tabular: List) -> None:
        super(TabularTransformerEncoder, self).__init__()

        self.num_cat = len(cat_lengths_tabular)
        self.num_con = len(con_lengths_tabular)
        self.num_unique_cat= sum(cat_lengths_tabular)
        print('TabularTransformerEncoder uses Mask Attention')
        # print('TabularTransformerEncoder No Mask Attention')

        # categorical embedding
        cat_offsets = torch.tensor([0] + cat_lengths_tabular[:-1]).cumsum(0)
        self.register_buffer('cat_offsets', cat_offsets, persistent=False)
        self.cat_embedding = nn.Embedding(self.num_unique_cat, args.tabular_embedding_dim)
        # continuous embedding
        self.con_proj = nn.Linear(1, args.tabular_embedding_dim)
        # class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, args.tabular_embedding_dim))
        self.mask_special_token = nn.Parameter(torch.zeros(1, 1, args.tabular_embedding_dim))
        pos_ids = torch.arange(self.num_cat+self.num_con+1).expand(1, -1)
        self.register_buffer('pos_ids', pos_ids, persistent=False)
        # print('TabularTransformerEncoder No Column Embedding')
        self.column_embedding = nn.Embedding(self.num_cat+self.num_con+1, args.tabular_embedding_dim)

        self.norm = nn.LayerNorm(args.tabular_embedding_dim)
        self.dropout = nn.Dropout(args.embedding_dropout) if args.embedding_dropout > 0. else nn.Identity()

        # transformer
        self.transformer_blocks = nn.ModuleList([
                            Block(dim=args.tabular_embedding_dim, drop=args.drop_rate, is_cross_attention=False) 
                            for i in range(args.tabular_transformer_num_layers)
                            ])

        
        if args.checkpoint is None:
            trunc_normal_(self.cls_token, std=.02)
            trunc_normal_(self.mask_special_token, std=.02)
            self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            m.weight.data.normal_(mean=0.0, std=.02)
        elif isinstance(m, nn.LayerNorm):
            m.bias.data.zero_()
            m.weight.data.fill_(1.0)
        if isinstance(m, nn.Linear) and m.bias is not None:
            m.bias.data.zero_()

    def embedding(self, x, mask_special=None):
        # categorical embedding
        # print(x[:, :self.num_cat].long()+self.cat_offsets)
        cat_x = self.cat_embedding(x[:, :self.num_cat].long()+self.cat_offsets)
        # continuous embedding
        con_x = self.con_proj(x[:, self.num_cat:].unsqueeze(-1))
        x = torch.cat([cat_x, con_x], dim=1)
        # mask special token
        if mask_special is not None:
            mask_special = mask_special.unsqueeze(-1)
            mask_special_tokens = self.mask_special_token.expand(x.shape[0], x.shape[1], -1)
            x = mask_special*mask_special_tokens + (~mask_special)*x
        # concat
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        column_embed = self.column_embedding(self.pos_ids)
        x = x+column_embed
        x = self.norm(x)
        x = self.dropout(x)
        return x
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor=None, mask_special: torch.Tensor=None) -> torch.Tensor:
        x = self.embedding(x, mask_special=mask_special)
        # create attention mask
        if mask is not None:
            B, N = mask.shape
            cls_mask = torch.zeros(B, 1).bool().to(mask.device)
            mask = torch.cat([cls_mask, mask], dim=1)
            mask = mask[:,None,:].repeat(1, N+1, 1)
            mask_eye = ~torch.eye(N+1).bool().to(mask.device)
            mask_eye = mask_eye[None,:,:]
            mask = mask * mask_eye
            mask = mask[:,None,:,:]
            mask = mask*(-1e9)
            assert x.shape[1] == mask.shape[2]
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, mask=mask)
            # x = transformer_block(x)
        return x


class MultimodalTransformerEncoder(nn.Module):
    '''
    Tabular Transformer Encoder based on BERT
    '''
    def __init__(self, args: Dict) -> None:
        super(MultimodalTransformerEncoder, self).__init__()
        self.image_proj = nn.Linear(args.embedding_dim, args.multimodal_embedding_dim)
        self.image_norm = nn.LayerNorm(args.multimodal_embedding_dim)
        self.tabular_proj = nn.Linear(args.tabular_embedding_dim, args.multimodal_embedding_dim) if args.tabular_embedding_dim != args.multimodal_embedding_dim else nn.Identity()
        self.transformer_blocks = nn.ModuleList([
                            Block(dim=args.multimodal_embedding_dim, is_cross_attention=True, 
                                  encoder_dim=args.multimodal_embedding_dim) 
                            for i in range(args.multimodal_transformer_num_layers)
                            ])
        self.norm =  nn.LayerNorm(args.multimodal_embedding_dim)

        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            m.weight.data.normal_(mean=0.0, std=.02)
        elif isinstance(m, nn.LayerNorm):
            m.bias.data.zero_()
            m.weight.data.fill_(1.0)
        if isinstance(m, nn.Linear) and m.bias is not None:
            m.bias.data.zero_()
    
    def forward(self, x: torch.Tensor, image_features: torch.Tensor, visualize=False) -> torch.Tensor:
        if len(image_features.shape) == 4:
            B,C,H,W = image_features.shape
            image_features = image_features.reshape(B,C,H*W).permute(0,2,1)
        image_features = self.image_proj(image_features)
        image_features = self.image_norm(image_features)
        x = self.tabular_proj(x)

        if visualize == False:
            for i, transformer_block in enumerate(self.transformer_blocks):
                x = transformer_block(x, encoder_hidden_states=image_features)
            x = self.norm(x)
            return x
        else:
            attns = []
            for i, transformer_block in enumerate(self.transformer_blocks):
                x, attn = transformer_block(x, encoder_hidden_states=image_features, visualize=visualize)
                attns.append(attn)
            x = self.norm(x)
            return x, attns



class TabularPredictor(nn.Module):
    '''Masked Tabular Reconstruction'''
    def __init__(self, args: Dict, cat_lengths_tabular: List, con_lengths_tabular: List, num_unique_cat: int=None) -> None:
        super(TabularPredictor, self).__init__()
        self.num_cat = len(cat_lengths_tabular)
        self.num_con = len(con_lengths_tabular)
        if num_unique_cat is None:
            self.num_unique_cat = sum(cat_lengths_tabular)
        else:
            self.num_unique_cat = num_unique_cat
        # categorical classifier
        self.cat_classifier = nn.Linear(args.tabular_embedding_dim, self.num_unique_cat, bias=True)
        # continuous regessor
        self.con_regressor = nn.Linear(args.tabular_embedding_dim, 1, bias=True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            m.weight.data.normal_(mean=0.0, std=.02)
        elif isinstance(m, nn.LayerNorm):
            m.bias.data.zero_()
            m.weight.data.fill_(1.0)
        if isinstance(m, nn.Linear) and m.bias is not None:
            m.bias.data.zero_()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # remove clstoken
        x = x[:,1:,:]
        # categorical classifier
        cat_x = self.cat_classifier(x[:, :self.num_cat])
        # continuous regessor
        con_x = self.con_regressor(x[:, self.num_cat:])
        return (cat_x, con_x)



if __name__ == "__main__":
    ### Attention Test
    # x = torch.randn(2, 5, 512)
    # mask = torch.tensor([[True, False, True, False, False],[False, True, False, False, False]])
    # B, N = mask.shape
    # mask = mask[:,None,:].repeat(1, N, 1)
    # mask_eye = ~torch.eye(N).bool()
    # mask_eye = mask_eye[None,:,:]
    # mask = mask * mask_eye
    # mask = mask[:,None,:,:]
    # mask = mask*(-1e9)
    # model = Attention(dim=512, num_heads=8)
    # out = model(x, mask=mask)
    # print(out.shape)

    # #### Cross Attention Test
    # x = torch.randn(2, 10, 512)
    # encoder_hidden_states = torch.randn(2, 12, 512)
    # model = Block(dim=512, num_heads=8, is_cross_attention=True, encoder_dim=512)
    # out = model(x, encoder_hidden_states)
    # print(out.shape)

    ### Tabular Transformer Encoder Test
    cat_lengths_tabular, con_lengths_tabular = [5,4,2], [1,1]
    args = DotDict({'tabular_embedding_dim': 512, 'tabular_transformer_num_layers': 4, 'embedding_dropout': 0.1,
                    'embedding_dim': 2048, 'drop_rate': 0.0,
                    'multimodal_embedding_dim': 512, 'multimodal_transformer_num_layers': 4})
    model = TabularTransformerEncoder(args, cat_lengths_tabular, con_lengths_tabular)
    x = torch.tensor([[4.0, 3.0, 0.0, 0.2, -0.1],
                     [2.0, 1.0, 1.0, -0.5, 0.2]], dtype=torch.float32)
    mask = torch.tensor([[True, True, False, False, False],[True, True, False, False, False]])
    mask_special = torch.tensor([[True, False, False, False, False],[False, True, False, False, False]])
    out = model(x, mask=mask, mask_special=mask_special)
    print(out.shape)    

    # ### Multimodal Transformer Encoder Test
    # x = torch.randn(2, 6, 512)
    # image_features = torch.randn(2, 49, 2048)
    # model = MultimodalTransformerEncoder(args)
    # out = model(x, image_features)
    # print(out.shape)
    # print(out)

    # ### Tabular Predictor Test
    # multimodal_features = torch.randn(2, 6, 512)
    # model = TabularPredictor(args, cat_lengths_tabular, con_lengths_tabular)
    # out = model(multimodal_features)
    # for y in out:
    #     print(y.shape)
    #     print(y)  

