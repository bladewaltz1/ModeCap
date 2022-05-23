import torch
import torch.nn.functional as F
from torch import nn

import numpy as np
from scipy.optimize import linear_sum_assignment
from transformers.models.bert.modeling_bert import (BertEncoder, 
                                                    BertPredictionHeadTransform,
                                                    BertLMPredictionHead,
                                                    BertEmbeddings)
from transformers.models.bert.configuration_bert import BertConfig

from utils.tokenizer import BOS, CLS, PAD


class Embeddings(nn.Module):
    def __init__(self, bert_cfg):
        super().__init__()
        self.word_embeddings = nn.Embedding(bert_cfg.vocab_size, 
                                            bert_cfg.hidden_size, 
                                            padding_idx=PAD)
        self.position_embeddings = nn.Embedding(512, bert_cfg.hidden_size)
        self.LayerNorm = nn.LayerNorm(bert_cfg.hidden_size,
                                       eps=bert_cfg.layer_norm_eps)
        self.dropout = nn.Dropout(bert_cfg.hidden_dropout_prob)

        self.register_buffer(
            "position_ids", 
            torch.arange(bert_cfg.max_position_embeddings).expand((1, -1))
        )

    def forward(self, input_ids, mode_embeds):
        position_ids = self.position_ids[:, :input_ids.size(1)]
        position_embeds = self.position_embeddings(position_ids)
        inputs_embeds = self.word_embeddings(input_ids)
        embeddings = inputs_embeds + position_embeds + mode_embeds
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class ModeEncoder(nn.Module):
    def __init__(self, cfg, num_layers):
        super().__init__()
        bert_cfg = BertConfig(
            hidden_size=cfg.hidden_size,
            num_hidden_layers=num_layers,
            hidden_dropout_prob=cfg.hidden_dropout_prob,
        )
        self.embedding_layer = BertEmbeddings(bert_cfg)
        self.encoder = BertEncoder(bert_cfg)

        self.mode_transform = BertPredictionHeadTransform(bert_cfg)
        self.mode_linear = nn.Linear(cfg.hidden_size, cfg.hidden_size)

    def forward(self, token_ids):
        mode_token = token_ids.new_full([token_ids.size(0), 1], CLS)
        token_ids = torch.cat([mode_token, token_ids], dim=1)
        embeds = self.embedding_layer(token_ids)

        attn_mask = (token_ids != PAD).float()
        attn_mask = attn_mask[:, None, None, :]
        attn_mask = (1.0 - attn_mask) * -10000.0

        hidden_state = self.encoder(embeds, attn_mask).last_hidden_state
        mode_embed = self.mode_transform(hidden_state[:, 0, :])
        mode_embed = self.mode_linear(mode_embed)
        return mode_embed


class ImgEncoder(nn.Module):
    def __init__(self, cfg, num_layers):
        super().__init__()
        self.feat_embed = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, cfg.hidden_size),
            nn.Dropout(cfg.hidden_dropout_prob)
        )
        self.meta_embed = nn.Sequential(
            nn.Linear(6 + 1601, cfg.hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.hidden_dropout_prob)
        )
        self.layernorm = nn.LayerNorm(cfg.hidden_size)
        self.dropout = nn.Dropout(cfg.hidden_dropout_prob)

        bert_cfg = BertConfig(
            hidden_size=cfg.hidden_size,
            num_hidden_layers=num_layers,
            hidden_dropout_prob=cfg.hidden_dropout_prob,
        )
        self.encoder = BertEncoder(bert_cfg)

        self.output_transform = BertPredictionHeadTransform(bert_cfg)
        self.output_linear = nn.Linear(cfg.hidden_size, cfg.hidden_size)

    def forward(self, region_feats, region_meta):
        region_feats = self.feat_embed(region_feats)
        region_meta = self.meta_embed(region_meta)
        img_feat = self.layernorm(region_feats + region_meta)
        img_feat = self.dropout(img_feat)

        hidden_state = self.encoder(img_feat).last_hidden_state
        hidden_state = self.output_transform(hidden_state)
        hidden_state = self.output_linear(hidden_state)
        return hidden_state


class Decoder(nn.Module):
    def __init__(self, cfg, num_layers):
        super().__init__()
        bert_cfg = BertConfig(
            hidden_size=cfg.hidden_size,
            hidden_dropout_prob=cfg.hidden_dropout_prob,
            num_hidden_layers=num_layers,
            is_decoder=True,
            add_cross_attention=True,
        )
        self.embedding_layer = Embeddings(bert_cfg)
        self.encoder = BertEncoder(bert_cfg)
        self.classifier = BertLMPredictionHead(bert_cfg)

        position_ids = torch.arange(bert_cfg.max_position_embeddings)
        attn_mask = position_ids[None, None, :].repeat(
            1, bert_cfg.max_position_embeddings, 1
        ) <= position_ids[None, :, None]
        self.register_buffer('attn_mask', attn_mask)

    def forward(self, img_feat, token_ids, mode_embed):
        batch_size = token_ids.size(0)

        start_token = token_ids.new_full([batch_size, 1], BOS)
        token_ids = torch.cat([start_token, token_ids], dim=1)
        embeds = self.embedding_layer(token_ids, mode_embed)

        seq_len = token_ids.size(1)
        attn_mask = self.attn_mask[:, :seq_len, :seq_len]
        attn_mask = attn_mask.repeat(batch_size, 1, 1)
        attn_mask = attn_mask[:, None, :, :].float()
        attn_mask = (1.0 - attn_mask) * -10000.0

        hidden_state = self.encoder(embeds, 
                                    attention_mask=attn_mask, 
                                    encoder_hidden_states=img_feat)

        preds = self.classifier(hidden_state.last_hidden_state)
        return preds


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, weight=None, smoothing=0.0):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.classes = classes
        self.weight = weight

    def forward(self, pred, target):
        """
        Args:
            pred: (N, C), float
            target: (N,), long, values in [0, C-1]
        """
        if self.weight is None:
            self.weight = torch.ones(self.classes, dtype=torch.float32,
                                     device=target.device)
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)

        weight = self.weight[target]
        weighted_loss = torch.sum(-true_dist * pred, dim=-1) * weight

        return torch.mean(weighted_loss) * weight.numel() / weight.sum()


class CodeBook(nn.Module):
    def __init__(self, cfg):
        super(CodeBook, self).__init__()
        self.embedding = nn.Embedding(cfg.num_modes, cfg.hidden_size)
        self.commitment_cost = cfg.loss.commitment_cost

    def forward(self, mode_emb, splits):
        distances = (torch.sum(mode_emb**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(mode_emb, self.embedding.weight.t())).sqrt()

        distances = distances.split(splits)
        indices = [
            linear_sum_assignment(d.detach().cpu().numpy())[1]
            for d in distances
        ]
        indices = torch.from_numpy(np.concatenate(indices))
        indices = indices.to(mode_emb.device)
        # print("indices: ", indices.unique())
        quantized = self.embedding(indices)

		# Loss
        q_latent_loss = F.mse_loss(mode_emb.detach(), quantized) + \
                        F.mse_loss(mode_emb.mean(dim=0).detach(), 
                                   self.embedding.weight.mean(dim=0))
        e_latent_loss = F.mse_loss(mode_emb, quantized.detach()) + \
                        F.mse_loss(mode_emb.mean(dim=0), 
                                   self.embedding.weight.mean(dim=0).detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized = mode_emb + (quantized - mode_emb).detach()

        return loss, quantized[:, None, :], indices
