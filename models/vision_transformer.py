import ml_collections
import copy
import math
import os

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
import numpy as np

logger = logging.getLogger(__name__)

class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        self.num_attention_heads = config.transformer.num_heads
        self.attention_head_size = config.embed_dim // self.num_attention_heads
        self.all_head_size       = self.num_attention_heads * self.attention_head_size

        self.query  = Linear(config.embed_dim, self.all_head_size)
        self.key    = Linear(config.embed_dim, self.all_head_size)
        self.value  = Linear(config.embed_dim, self.all_head_size)
        self.out    = Linear(config.embed_dim, config.embed_dim)

        self.attn_drop = Dropout(config.transformer.attention_dropout_rate)
        self.proj_drop = Dropout(config.transformer.attention_dropout_rate)

        self.softmax = Softmax(dim=-1)
    
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, hidden_states):
        mixed_query_layer   = self.query(hidden_states)
        mixed_key_layer     = self.key(hidden_states)
        mixed_value_layer   = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer   = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        weights = attention_probs  = self.softmax(attention_scores)
        attention_probs = self.attn_drop(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        attention_output = self.out(context_layer)
        attention_output = self.proj_drop(attention_output)
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.embed_dim, config.transformer.mlp_dim)
        self.act = nn.GELU()
        self.fc2 = Linear(config.transformer.mlp_dim, config.embed_dim)
        self.dropout = Dropout(config.transformer.dropout_rate)

        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings
    """
    def __init__(self, config, image_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        image_size = _pair(image_size)
        patch_size = _pair(config.patches.size)

        if config.split == 'non-overlap':
            n_patches = (image_size[0] // patch_size[0]) * (image_size[1] // patch_size[1])
            self.patch_embeddings = Conv2d(in_channels=in_channels, out_channels=config.embed_dim, kernel_size=patch_size, stride=patch_size)
        elif config.split == 'overlap':
            n_patches = ((image_size[0] - patch_size[0]) // config.slide_step + 1) * ((image_size[1] - patch_size[1]) // config.slide_step + 1)
            self.patch_embeddings = Conv2d(in_channels=in_channels, out_channels=config.embed_dim, kernel_size=patch_size, stride=config.slide_step)

        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches + 1, config.embed_dim)) #(1, 197, 768)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim)) #(1, 1, 768)

        self.dropout = Dropout(config.transformer.dropout_rate)
    
    def forward(self, x):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)

        if self.hybrid:
            x = self.hybrid_model(x)
        
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        x = torch.cat((cls_tokens, x), dim=1)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class Block(nn.Module):
    def __init__(self, config):
        super(Block, self).__init__()
        self.embed_dim      = config.embed_dim
        self.attention_norm = LayerNorm(config.embed_dim, eps=1e-6)
        self.attn           = Attention(config)
        self.mlp_norm       = LayerNorm(config.embed_dim, eps=1e-6)
        self.mlp            = Mlp(config)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = x + h
        return x, weights
    
    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight    = np2th(weights[os.path.join(ROOT, "MultiHeadDotProductAttention_1/query",  "kernel")]).view(self.embed_dim, self.embed_dim).t()
            key_weight      = np2th(weights[os.path.join(ROOT, "MultiHeadDotProductAttention_1/key",    "kernel")]).view(self.embed_dim, self.embed_dim).t()
            value_weight    = np2th(weights[os.path.join(ROOT, "MultiHeadDotProductAttention_1/value",  "kernel")]).view(self.embed_dim, self.embed_dim).t()
            out_weight      = np2th(weights[os.path.join(ROOT, "MultiHeadDotProductAttention_1/out",    "kernel")]).view(self.embed_dim, self.embed_dim).t()

            query_bias      = np2th(weights[os.path.join(ROOT, "MultiHeadDotProductAttention_1/query",  "bias")]).view(-1)
            key_bias        = np2th(weights[os.path.join(ROOT, "MultiHeadDotProductAttention_1/key",    "bias")]).view(-1)
            value_bias      = np2th(weights[os.path.join(ROOT, "MultiHeadDotProductAttention_1/value",  "bias")]).view(-1)
            out_bias        = np2th(weights[os.path.join(ROOT, "MultiHeadDotProductAttention_1/out",    "bias")]).view(-1)

            self.attn.query.weight. copy_(query_weight)
            self.attn.key.weight.   copy_(key_weight)
            self.attn.value.weight. copy_(value_weight)
            self.attn.out.weight.   copy_(out_weight)

            self.attn.query.bias.copy_(query_bias)
            self.attn.key.  bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.  bias.copy_(out_bias)

            mlp_weight_0    = np2th(weights[os.path.join(ROOT, "MlpBlock_3/Dense_0", "kernel")]).t()
            mlp_weight_1    = np2th(weights[os.path.join(ROOT, "MlpBlock_3/Dense_1", "kernel")]).t()
            mlp_bias_0      = np2th(weights[os.path.join(ROOT, "MlpBlock_3/Dense_0", "bias")]).t()
            mlp_bias_1      = np2th(weights[os.path.join(ROOT, "MlpBlock_3/Dense_1", "bias")]).t()

            self.mlp.fc1.weight .copy_(mlp_weight_0)
            self.mlp.fc2.weight .copy_(mlp_weight_1)
            self.mlp.fc1.bias   .copy_(mlp_bias_0)
            self.mlp.fc2.bias   .copy_(mlp_bias_1)

            self.attention_norm.weight  .copy_(np2th(weights[os.path.join(ROOT, "LayerNorm_0", "scale")]))
            self.attention_norm.bias    .copy_(np2th(weights[os.path.join(ROOT, "LayerNorm_0", "bias")]))

            self.mlp_norm.weight        .copy_(np2th(weights[os.path.join(ROOT, "LayerNorm_2", "scale")]))
            self.mlp_norm.bias          .copy_(np2th(weights[os.path.join(ROOT, "LayerNorm_2", "bias")]))


class Part_Attention(nn.Module):
    def __init__(self):
        super(Part_Attention, self).__init__()
    
    def forward(self, x):
        length = len(x)
        last_map = x[0]
        for i in range(1, length):
            last_map = torch.matmul(x[i], last_map)
        last_map = last_map[:, :, 0, 1:]

        _, max_idx = last_map.max(2)
        return _, max_idx

class EncoderPSM(nn.Module):
    """ TransFG: Part Select Module
    """
    def __init__(self, config):
        super(EncoderPSM, self).__init__()
        self.layers = nn.ModuleList()
        layer = Block(config)
        for _ in range(config.transformer.depth - 1):
            self.layers.append(copy.deepcopy(layer))
        
        self.part_select    = Part_Attention()
        self.part_layer     = Block(config)
        self.part_norm      = LayerNorm(config.embed_dim, eps=1e-6)
    
    def forward(self, hidden_states):
        attn_weights = []
        for layer in self.layers:
            hidden_states, weights = layer(hidden_states)
            attn_weights.append(weights)
        part_num, part_idx = self.part_select(attn_weights)
        part_idx += 1

        parts = []
        B, num = part_idx.shape
        for i in range(B):
            parts.append(hidden_states[i, part_idx[i, :]])
        parts = torch.stack(parts).squeeze(1)
        concat = torch.cat((hidden_states[:,0].unsqueeze(1), parts), dim=1)

        part_states, _ = self.part_layer(concat)

        part_encoded = self.part_norm(part_states)
        return part_encoded

    
class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList()
        layer = Block(config)
        for _ in range(config.transformer.depth):
            self.layers.append(copy.deepcopy(layer))
        self.part_norm = LayerNorm(config.embed_dim, eps=1e-6)
    
    def forward(self, hidden_states):
        for layer in self.layers:
            hidden_states, weights = layer(hidden_states)
        return self.part_norm(hidden_states)


class Transformer(nn.Module):
    def __init__(self, config, image_size):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, image_size=image_size)
        self.encoder = EncoderPSM(config) if config.mode == 'psm' else Encoder(config)
        
        # Representation Layer
        if config.representation_size:
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(config.embed_dim, config.representation_size)),
                ('act', nn.Tanh()),
            ]))
        else:
            self.pre_logits = nn.Identity()
    
    def forward(self, input_ids):
        embedding_output = self.embeddings(input_ids)
        part_encoded = self.encoder(embedding_output)
        return part_encoded


class VisionTransformer(nn.Module):
    def __init__(self, config, image_size=224, num_classes=1000, zero_head=False):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.num_features = config.representation_size or config.embed_dim
        self.num_tokens = 2 if config.distilled else 1
        self.zero_head = zero_head
        self.classifier = config.classifier

        self.transformer = Transformer(config, image_size)
        self.head = Linear(config.embed_dim, num_classes)
        if config.distilled:
            self.head_dist = Linear(config.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()
        else:
            self.head_dist = None
        
    
    def forward_features(self, x):
        part_tokens = self.transformer(x)
        return part_tokens[:, 0]
    
    def forward(self, x):
        x = self.transformer(x)
        logits = self.head(x)
        return logits
        
    def load_from(self, weights):
        with torch.no_grad():
            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))
            self.transformer.embeddings.cls_token.copy_(np2th(weights["cls"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resize variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)

                if self.classifier == "token":
                    posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
                    ntok_new -= 1
                else:
                    posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
            
                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = np.concatenate([posemb_tok, posemb_grid], axis=1)
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))
            
            self.transformer.encoder.part_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.part_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            for bname, block in self.transformer.encoder.named_children():
                if bname.startswith('part') == False:
                    for uname, unit in block.named_children():
                        unit.load_from(weights, n_block=uname)
            
            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(weights["conv_root/kernel"], conv=True))
                gn_weight = np2th(weights["gn_root/scale"]).view(-1)
                gn_bias = np2th(weights["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(weights, n_block=bname, n_unit=uname)

def con_loss(features, labels):
    B, _ = features.shape
    pos_label_matrix = torch.stack([labels == labels[i] for i in range(B)]).float()
    neg_label_matrix = 1 - pos_label_matrix

    features = F.normalize(features)
    cos_matrix = features.mm(features.t())
    pos_cos_matrix = 1 - cos_matrix
    neg_cos_matrix = cos_matrix - 0.4
    neg_cos_matrix[neg_cos_matrix < 0] = 0
    loss = (pos_cos_matrix * pos_label_matrix).sum() + (neg_cos_matrix * neg_label_matrix).sum()
    loss /= (B* B)
    return loss


def get_b16_config():
    """Returns the ViT-B/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches      = ml_collections.ConfigDict({'size': (16, 16)})
    config.split        = 'non-overlap'
    config.slide_step   = 12
    config.embed_dim    = 768

    config.transformer              = ml_collections.ConfigDict()
    config.transformer.mlp_dim      = 3072
    config.transformer.num_heads    = 12
    config.transformer.depth        = 12

    config.transformer.attention_dropout_rate   = 0.0
    config.transformer.dropout_rate             = 0.1

    config.classifier           = 'token'
    config.representation_size  = None # enable and set representation layer (pre-logits) to this value if set
    config.distilled            = False # model includes a distillation token and head as in DeiT models
    config.mode = 'vit'
    return config

def get_b32_config():
    config              = get_b16_config()
    config.patches.size = (32, 32)
    return config

def get_b16_psm_config():
    config      = get_b16_config()
    config.mode = 'psm'
    return config

def get_b32_psm_config():
    config              = get_b32_config()
    config.patches.size = (32, 32)
    config.mode         = 'psm'
    return config

def get_testing():
    """Returns a minimal configuration for testing."""
    config = get_b16_config()
    config.embed_dim               = 1
    config.transformer.mlp_dim      = 1
    config.transformer.num_heads    = 1
    config.transformer.depth   = 1
    return config

def get_config(module_name):
    return {
        'ViT-Test'      : get_testing(),
        'ViT-B_16'      : get_b16_config(),
        'ViT-B_32'      : get_b32_config(),
        'ViT-B_16_PSM'  : get_b16_psm_config(),
        'ViT-B_32_PSM'  : get_b32_psm_config(),
    }[module_name]

def np2th(weights, conv=False):
    if weights.ndim == 4 and weights.shape[0] == weights.shape[1] == weights.shape[2] == 1:
        weights = weights.flatten()
    if conv:
        if weights.ndim == 4:
            weights = weights.transpose([3, 2, 0, 1])
        elif weights.ndim == 3:
            weights = weights.transpose([2, 0, 1])
        elif weights.ndim == 2:
            weights = weights.transpose([1, 0])
    return torch.from_numpy(weights)
