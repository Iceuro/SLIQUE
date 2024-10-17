
import numpy as np
# import timm
import torch
from torch import nn

# import losses
from collections import OrderedDict
from typing import Tuple, Union
from torch.utils.data import Dataset, DataLoader
import argparse
import torch
import clip
import logging
import torch.nn.functional as F
from torch import nn, optim
# import pandas as pd
from PIL import Image
# import natsort
import os
from tqdm import tqdm
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 vision_width: int,
                 vision_model: nn.Module,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 **kwargs,
                 ):
        super().__init__()

        self.context_length = context_length
        self.vision_width = vision_width

        self.visual = vision_model

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.image_projection = nn.Parameter(torch.empty(vision_width, embed_dim))
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        nn.init.normal_(self.image_projection, std=self.vision_width ** -0.5)
        nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        x = self.visual(image)
        x = x @ self.image_projection

        return x

    def encode_text(self, text):
        x = self.token_embedding(text)
        # x = x.unsqueeze(0)# [batch_size, n_ctx, d_model]  #test 加
        x = x + self.positional_embedding

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text

class CONTRIQUE_model(nn.Module):
    # resnet50 architecture with projector
    def __init__(self, args, encoder, n_features, \
                 patch_dim=(2, 2), normalize=True, projection_dim=128):
        super(CONTRIQUE_model, self).__init__()

        self.normalize = normalize
        self.encoder = nn.Sequential(*list(encoder.children())[:-2])
        self.n_features = 768
        self.n_features1 = 196
        self.patch_dim = patch_dim

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool_patch = nn.AdaptiveAvgPool2d(patch_dim)

        # MLP for projector
        self.projector = nn.Sequential(
            nn.Linear(self.n_features, self.n_features, bias=False),
            nn.BatchNorm1d(self.n_features),
            nn.ReLU(),
            nn.Linear(self.n_features, projection_dim, bias=False),
            nn.BatchNorm1d(projection_dim),
        )

    def forward(self, x_i, x_j):
        # global features
        h_i = self.encoder(x_i)
        h_j = self.encoder(x_j)

        # local features
        h_i_patch = h_i
        h_j_patch = h_j

        # h_i_patch = h_i_patch.reshape(-1, self.n_features, \
        #                               self.patch_dim[0] * self.patch_dim[1])
        #
        # h_j_patch = h_j_patch.reshape(-1, self.n_features, \
        #                               self.patch_dim[0] * self.patch_dim[1])
        #
        # h_i_patch = torch.transpose(h_i_patch, 2, 1)
        # h_i_patch = h_i_patch.reshape(-1, self.n_features)
        #
        # h_j_patch = torch.transpose(h_j_patch, 2, 1)
        # h_j_patch = h_j_patch.reshape(-1, self.n_features)

        # h_i = self.avgpool(h_i)
        # h_j = self.avgpool(h_j)
        #
        # h_i = h_i.view(-1, self.n_features)
        # h_j = h_j.view(-1, self.n_features)

        if self.normalize:
            h_i = nn.functional.normalize(h_i, dim=1)
            h_j = nn.functional.normalize(h_j, dim=1)

            # h_i_patch = nn.functional.normalize(h_i_patch, dim=1)
            # h_j_patch = nn.functional.normalize(h_j_patch, dim=1)

        # global projections
        z_i = self.projector(h_i)
        z_j = self.projector(h_j)

        # local projections
        z_i_patch =h_i_patch
        z_j_patch = h_j_patch

        return z_i, z_j, z_i_patch, z_j_patch, h_i, h_j, h_i_patch, h_j_patch


class CON_model(CLIP):
        # resnet50 architecture with projector
    # def __init__(self,  args,n_features, model,\
    #                   projection_dim=128, **kwargs):
    #         super().__init__(**kwargs)

    def __init__(self, args, n_features, \
                         patch_dim=(2,2), normalize=True, projection_dim=128,**kwargs):
            super().__init__(**kwargs)

            self.normalize = normalize
            self.encoder = nn.Sequential(*list(self.visual.children())[:-2])
            self.n_features = 768
            self.ssl_mlp_dim = args.ssl_mlp_dim
            self.patch_dim = patch_dim
            self.projection_dim=projection_dim
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.avgpool_patch = nn.AdaptiveAvgPool2d(patch_dim)

            # MLP for projector
            # self.projector = nn.Sequential(
            #
            #     nn.Linear(self.n_features, self.n_features, bias=False),
            #     nn.BatchNorm1d(self.n_features),
            #     nn.ReLU(),
            #     nn.Linear(self.n_features, 1, bias=False),
            #     nn.BatchNorm1d(1),
            # )

            # self.n_features = n_features

            # self.model = model

            self.conv1 = nn.Conv1d(in_channels=49 , out_channels=4, kernel_size=1)
            # self.conv1 = nn.Conv1d(in_channels=49 , out_channels=512, kernel_size=1)




    #
    def forward(self, x_i, x_j,y_i):

        h_i = self.encoder(x_i)
        h_j = self.encoder(x_j)

        # local features
        # h_i_patch = self.avgpool_patch(h_i)
        # h_j_patch = self.avgpool_patch(h_j)
        #
        #
        if self.normalize:
           h_i = nn.functional.normalize(h_i, dim=1)
           h_j = nn.functional.normalize(h_j, dim=1)

        h_i = self.conv1(h_i)
        h_j = self.conv1(h_j)
        #
        #
        #
        # h_i_patch = h_i_patch.reshape(-1, self.n_features, \
        #                               self.patch_dim[0] * self.patch_dim[1])
        #
        # h_j_patch = h_j_patch.reshape(-1, self.n_features, \
        #                               self.patch_dim[0] * self.patch_dim[1])
        #
        # h_i_patch = torch.transpose(h_i_patch, 2, 1)
        # h_i_patch = h_i_patch.reshape(-1, self.n_features)
        #
        # h_j_patch = torch.transpose(h_j_patch, 2, 1)
        # h_j_patch = h_j_patch.reshape(-1, self.n_features)
        #
        # h_i = self.avgpool(h_i)
        # h_j = self.avgpool(h_j)
        #
        # h_i = h_i.view(-1, self.n_features)
        # h_j = h_j.view(-1, self.n_features)

        if self.normalize:
            h_i = nn.functional.normalize(h_i, dim=1)
            h_j = nn.functional.normalize(h_j, dim=1)

            # h_i_patch = nn.functional.normalize(h_i_patch, dim=1)
            # h_j_patch = nn.functional.normalize(h_j_patch, dim=1)


        # global projections
        # z_i = self.projector(h_i)
        # z_j = self.projector(h_j)
        z_i = h_i.reshape(-1, 512)
        z_j = h_j.reshape(-1, 512)
        # local projections
        # z_i_patch = self.projector(h_i_patch)
        # z_j_patch = self.projector(h_j_patch)


        image_embed = h_i


        text_embed =  self.encode_text(y_i)
        # text_embed1 =  self.encode_text1(y_i)
        image_embed = image_embed @ self.image_projection
        # image_embed=torch.unsqueeze(image_embed,dim=-1)
        image_embed =  self.conv1(image_embed)

        # text_embed=text_embed @ self.text_projection1
        # normalized features
        if_i = image_embed / image_embed.norm(dim=1, keepdim=True)
        tf_i = text_embed / text_embed.norm(dim=1, keepdim=True)
        tf_i=tf_i.float()
        # tf_i = text_embed / text_embed.norm(dim=1, keepdim=True)

        if_i = if_i.reshape(-1, 512)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logit_scale = logit_scale.float()
        logits_per_image = logit_scale * if_i @ tf_i.t()
        logits_per_text = logits_per_image.t()


        return z_i, z_j, h_i, h_j, logits_per_image, logits_per_text


# class SLIP(CLIP):
#     def __init__(self,
#                  ssl_mlp_dim: int,
#                  ssl_emb_dim: int,
#                  **kwargs,
#                  ):
#         super().__init__(**kwargs)
#
#         self.image_mlp = self._build_mlp(in_dim=self.vision_width, mlp_dim=ssl_mlp_dim, out_dim=ssl_emb_dim)
#
#     def _build_mlp(self, in_dim, mlp_dim, out_dim):
#         return nn.Sequential(OrderedDict([
#             ("layer1", nn.Linear(in_dim, mlp_dim)),
#             # ("bn1", nn.SyncBatchNorm(mlp_dim)),
#             ("bn1", nn.BatchNorm1d(mlp_dim)),
#             ("relu1", nn.ReLU(inplace=True)),
#             ("layer2", nn.Linear(mlp_dim, mlp_dim)),
#             # ("bn2", nn.SyncBatchNorm(mlp_dim)),
#             ("bn2", nn.BatchNorm1d(mlp_dim)),
#             ("relu2", nn.ReLU(inplace=True)),
#             ("layer3", nn.Linear(mlp_dim, out_dim)),
#         ]))
#
#     def forward(self, image, text, aug1, aug2):
#         aug1_embed = self.image_mlp(self.visual(aug1))
#         aug2_embed = self.image_mlp(self.visual(aug2))
#
#         image_embed = self.encode_image(image)
#         text_embed = self.encode_text(text)
#
#         return {'image_embed': image_embed,
#                 'text_embed': text_embed,
#                 'logit_scale': self.logit_scale.exp(),
#                 'aug1_embed': aug1_embed,
#                 'aug2_embed': aug2_embed}
