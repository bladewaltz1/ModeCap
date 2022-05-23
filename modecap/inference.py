import argparse
import json
import logging
import os
from tqdm import tqdm

import torch
from torch import nn

from utils import mkdir
from utils.checkpointer import Checkpointer
from utils.dataloader import make_data_loader
from utils.logger import setup_logger
from utils.tokenizer import BOS, tokenizer

from modecap10a.config import _C as cfg
from modecap10a.dataset import CaptionDataset, collate_fn
from modecap10a.model import CodeBook, Decoder, ImgEncoder


def test(cfg, modules, data_loader, device):
    logger = logging.getLogger('test')
    logger.info('Start testing')
    modules.eval()

    modes = torch.tensor(range(64))
    modes = modes.to(torch.int32).to(device)
    # mode encoder
    quantized = modules['codebook'].embedding(modes)
    avg_mode = modules['codebook'].embedding.weight.mean(dim=0)[None]
    quantized = torch.cat([quantized, avg_mode], dim=0)
    modes = torch.cat([modes, -1 * modes.new_ones([1])])

    captions = {}
    for iteration, batch in tqdm(enumerate(data_loader, 0), 
                                 total=len(data_loader)):
        iteration = iteration + 1

        # data processing
        region_feats = batch[2].to(device)  # (N, S, 2048), float
        region_cls = batch[3].to(device)  # (N, D, 1601), float
        region_loc = batch[4].to(device)  # (N, D, 6), float

        region_loc[:, :, [0, 2]] /= region_loc[:, :, [2]] + 1e-5
        region_loc[:, :, [1, 3]] /= region_loc[:, :, [3]] + 1e-5
        rel_area = (region_loc[:, :, [3]] - region_loc[:, :, [1]]) * \
                   (region_loc[:, :, [2]] - region_loc[:, :, [0]])
        region_loc = torch.cat((region_loc[:, :, :4],
            rel_area.clamp_(0), region_loc[:, :, 5:]), dim=-1)
        region_meta = torch.cat([region_loc, region_cls], dim=-1)

        # image encoder
        img_emb = modules['img_encoder'](region_feats, region_meta)

        for mode_id, quant in zip(modes, quantized):
            # autoregressive decoding
            batch_size = img_emb.size(0)
            current_token_ids = modes.new_full([batch_size, 1], BOS)
            for _ in range(cfg.num_positions):
                current_embed = modules['decoder'].embedding_layer(
                    current_token_ids, quant[None, None, :]
                )

                seq_len = current_token_ids.size(1)
                attn_mask = modules['decoder'].attn_mask[:, :seq_len, :seq_len]
                attn_mask = attn_mask.repeat(batch_size, 1, 1)
                attn_mask = attn_mask[:, None, :, :].float()
                attn_mask = (1.0 - attn_mask) * -10000.0

                output = modules['decoder'].encoder(
                    current_embed, 
                    attention_mask=attn_mask, 
                    encoder_hidden_states=img_emb
                )
                hidden_state = output.last_hidden_state[:, [-1], :]
                preds = modules['decoder'].classifier(hidden_state)
                next_token_id = preds.max(dim=2)[1]
                current_token_ids = torch.cat([current_token_ids, 
                                               next_token_id], dim=1)

            for i, caption in enumerate(current_token_ids):
                caption = tokenizer.decode(caption, skip_special_tokens=False)
                caption = caption.split('[unused3]')[0][9:].strip()
                if batch[6][i] not in captions.keys():
                    captions[batch[6][i]] = [{'caption': caption, 
                                              'mode': int(mode_id.cpu())}]
                else:
                    captions[batch[6][i]].append({'caption': caption, 
                                                  'mode': int(mode_id.cpu())})

    return captions


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg.merge_from_list(args.opts)
    cfg.freeze()

    model_name = cfg.model_path.split('/')[-1].split('.')[0]
    save_dir = os.path.join(cfg.save_dir, f'test_{model_name}')
    mkdir(save_dir)
    logger = setup_logger('test', save_dir, 0)
    logger.info('Running with cfg:\n{}'.format(cfg))
    device = torch.device(cfg.device)

    modules = nn.ModuleDict({
        'codebook': CodeBook(cfg),
        'img_encoder': ImgEncoder(cfg, cfg.num_img_encoder_layers),
        'decoder': Decoder(cfg, cfg.num_decoder_layers)
    })
    modules = modules.to(device)

    checkpointer = Checkpointer(
        model=modules,
        save_dir=save_dir,
        save_to_disk=True,
        logger=logger,
    )
    checkpointer.load(cfg.model_path, is_strict=False, model_only=True)

    dataset = CaptionDataset(
        root=cfg.data_dir,
        split='test',
        max_length=cfg.num_positions,
    )

    data_loader = make_data_loader(
        dataset=dataset,
        collate_fn=collate_fn,
        batch_size=cfg.samples_per_gpu,
        num_workers=cfg.num_workers,
        split='test',
    )

    with torch.no_grad():
        captions = test(cfg, modules, data_loader, device)
    with open(os.path.join(save_dir, 'captions.json'), 'w') as f:
        json.dump(captions, f)
