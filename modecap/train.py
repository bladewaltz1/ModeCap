import argparse
import itertools
import logging
import os
import time

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils import clip_grad_norm_

import transformers
from transformers.optimization import AdamW

from utils import get_rank, mkdir, synchronize
from utils.checkpointer import Checkpointer
from utils.dataloader import make_data_loader
from utils.logger import setup_logger
from utils.tokenizer import MASK, PAD, num_tokens

from modecap.config import _C as cfg
from modecap.dataset import CaptionDataset, collate_fn
from modecap.model import (ImgEncoder, ModeEncoder, Decoder,
                           LabelSmoothingLoss, CodeBook)


def train(cfg, modules, criterion, optimizer, data_loader, 
          scheduler, checkpointer, device, arguments):
    logger = logging.getLogger('train')
    logger.info('Start training')
    max_iter = len(data_loader)
    start_iter = arguments['iteration']
    modules.train()

    end = time.time()
    for iteration, batch in enumerate(data_loader, start_iter):
        iteration = iteration + 1
        arguments['iteration'] = iteration

        # data processing
        token_ids = batch[0].to(device)  # (5N, L), long
        target_ids = batch[1].to(device)  # (5N, L + 1), long
        region_feats = batch[2].to(device)  # (N, S, 2048), float
        region_cls = batch[3].to(device)  # (N, D, 1601), float
        region_loc = batch[4].to(device)  # (N, D, 6), float

        img_ids = [[i] * count for i, count in enumerate(batch[5])]
        img_ids = list(itertools.chain(*img_ids))

        region_loc[:, :, [0, 2]] /= region_loc[:, :, [2]] + 1e-5
        region_loc[:, :, [1, 3]] /= region_loc[:, :, [3]] + 1e-5
        rel_area = (region_loc[:, :, [3]] - region_loc[:, :, [1]]) * \
                   (region_loc[:, :, [2]] - region_loc[:, :, [0]])
        region_loc = torch.cat((region_loc[:, :, :4],
            rel_area.clamp_(0), region_loc[:, :, 5:]), dim=-1)
        region_meta = torch.cat([region_loc, region_cls], dim=-1)

        # mode encoder
        mode = modules['mode_encoder'](token_ids)
        loss_vq, quant, _ = modules['codebook'](mode, batch[5])

        # image encoder
        img_emb = modules['img_encoder'](region_feats, region_meta)
        img_emb = img_emb[img_ids] # 5N, S, D

        # mask decoder
        unpad_positions = target_ids != PAD
        mask_token_ids = token_ids.new_full(token_ids.shape, MASK)
        img_emb_ = img_emb.detach()
        pred_logits = modules['mask_decoder'](img_emb_, mask_token_ids, quant)
        loss_mask = criterion['XE'](pred_logits[unpad_positions], 
                                    target_ids[unpad_positions])

        # decoder
        rand_idx = []
        cumsum = 0
        for size in batch[5]:
            rand_idx.append(torch.rand([size]).topk(1)[1] + cumsum)
            cumsum += size
        rand_idx = torch.cat(rand_idx, dim=0).to(cfg.device)

        img_emb_ = img_emb[rand_idx]
        token_ids_ = token_ids[rand_idx]
        quant_ = quant[rand_idx]
        pred_logits = modules['decoder'](img_emb_, token_ids_, quant_)

        target_ids_ = target_ids[rand_idx]
        unpad_positions = target_ids_ != PAD
        pred_logits = pred_logits[unpad_positions]
        target_ids_ = target_ids_[unpad_positions]
        loss_cap1 = criterion['XE'](pred_logits, target_ids_)
        pred_ids = pred_logits.max(dim=-1)[1]
        acc = (pred_ids == target_ids_).float().sum() / len(pred_ids)

        avg_mode = modules['codebook'].embedding.weight.mean(dim=0)
        pred_logits = modules['decoder'](img_emb_, token_ids_, avg_mode)
        pred_logits = pred_logits[unpad_positions]
        loss_cap2 = criterion['XE'](pred_logits, target_ids_)
        loss_cap = (loss_cap1 + loss_cap2) / 2

        # training
        optimizer.zero_grad()
        loss = loss_mask + loss_cap + loss_vq
        loss.backward()
        clip_grad_norm_(modules.parameters(), cfg.solver.grad_clip)
        optimizer.step()
        scheduler.step()
        batch_time = time.time() - end
        end = time.time()

        # logging
        if iteration % cfg.log_time == 0 or iteration == max_iter:
            logger.info(
                '  '.join([
                    'iter: {iter}', 'time: {time:.4f}', 'lr: {lr:.8f}',
                    'loss_mask: {loss_mask:.4f}', 'loss_cap: {loss_cap:.4f}',
                    'loss_vq: {loss_vq:.4f}', 'acc: {acc:.4f}'
                ]).format(
                    iter=iteration, time=batch_time, 
                    loss_mask=loss_mask, loss_cap=loss_cap, 
                    loss_vq=loss_vq, acc=acc,
                    lr=optimizer.param_groups[0]['lr'],
                ))
        if iteration % cfg.checkpoint_time == 0 or iteration == max_iter:
            checkpointer.save('model_{:07d}'.format(iteration), **arguments)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if cfg.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group('nccl', init_method='env://')
        synchronize()

    cfg.merge_from_list(args.opts)
    cfg.freeze()

    save_dir = os.path.join(cfg.save_dir, f'train')
    mkdir(save_dir)
    logger = setup_logger('train', save_dir, get_rank())
    logger.info('Running with cfg:\n{}'.format(cfg))

    arguments = {'iteration': 0}
    device = torch.device(cfg.device)

    modules = nn.ModuleDict({
        'codebook': CodeBook(cfg),
        'mode_encoder': ModeEncoder(cfg, cfg.num_mode_encoder_layers),
        'img_encoder': ImgEncoder(cfg, cfg.num_img_encoder_layers),
        'mask_decoder': Decoder(cfg, cfg.num_mask_decoder_layers),
        'decoder': Decoder(cfg, cfg.num_decoder_layers)
    })
    modules = modules.to(device)

    criterion = {'XE': LabelSmoothingLoss(num_tokens, 
                       smoothing=cfg.loss.label_smoothing)}

    optimizer = AdamW(
        params=modules.parameters(),
        lr=cfg.solver.lr,
        weight_decay=cfg.solver.weight_decay,
        betas=cfg.solver.betas
    )

    scheduler = transformers.get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=cfg.scheduler.warmup_steps,
        num_training_steps=cfg.scheduler.max_steps
    )

    checkpointer = Checkpointer(
        model=modules,
        optimizer=optimizer,
        scheduler=scheduler,
        save_dir=save_dir,
        save_to_disk=get_rank() == 0,
        logger=logger
    )

    if cfg.model_path is not None:
        extra_checkpoint_data = checkpointer.load(cfg.model_path)
        arguments.update(extra_checkpoint_data)

    dataset = CaptionDataset(
        root=cfg.data_dir,
        split='trainrestval',
        max_length=cfg.num_positions,
    )

    data_loader = make_data_loader(
        dataset=dataset,
        collate_fn=collate_fn,
        batch_size=cfg.samples_per_gpu,
        num_workers=cfg.num_workers,
        max_iter=cfg.scheduler.max_steps,
        split='trainrestval',
        is_distributed=cfg.distributed,
        start_iter=arguments['iteration'],
    )

    if cfg.distributed:
        modules = DistributedDataParallel(
            module=modules,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
        )

    train(cfg=cfg,
          modules=modules,
          criterion=criterion,
          optimizer=optimizer,
          data_loader=data_loader,
          scheduler=scheduler,
          checkpointer=checkpointer,
          device=device,
          arguments=arguments)
