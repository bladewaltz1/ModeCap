import json
import os
import h5py
import itertools

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from utils.tokenizer import EOS, PAD, tokenizer


feature_dir = 'region_feat_gvd_wo_bgd/feat_cls_1000'
feature_prefix = 'coco_detection_vg_100dets_gvd_checkpoint_trainval_'
box_dir = 'region_feat_gvd_wo_bgd'
box_file = 'coco_detection_vg_thresh0.2_feat_gvd_checkpoint_trainvaltest.h5'


class CaptionDataset(Dataset):
    def __init__(self, root, split, max_length):
        self.root = root
        self.split = split
        self.max_length = max_length

        with open(os.path.join(root, 'annotations', 'dataset_coco.json')) as f:
            database = json.load(f)
        database = database['images']

        self.database = []
        for item in database:
            if item['split'] in split:
                self.database.append(item)

    def get_region_feat(self, name):
        with h5py.File(os.path.join(self.root, feature_dir, 
                    f'{feature_prefix}feat{name[-3:]}.h5'), 'r') as features:
            region_feat = torch.from_numpy(features[name][:])

        with h5py.File(os.path.join(self.root, feature_dir,
                    f'{feature_prefix}cls{name[-3:]}.h5'), 'r') as classes:
            region_cls = torch.from_numpy(classes[name][:])

        with h5py.File(os.path.join(self.root, box_dir, box_file), 'r') as boxes:
            region_loc = torch.from_numpy(boxes[name][:])

        return region_feat, region_cls, region_loc

    def __getitem__(self, index):
        sample = self.database[index]
        captions = []
        captions_w_eos = []
        for item in sample['sentences']:
            caption = ' '.join(item['tokens'])
            caption = tokenizer.encode(caption, add_special_tokens=False)
            caption_w_eos = caption + [EOS]
            if len(caption) >= self.max_length:
                continue
            caption = torch.tensor(caption, dtype=torch.long)
            caption_w_eos = torch.tensor(caption_w_eos, dtype=torch.long)
            captions.append(caption)
            captions_w_eos.append(caption_w_eos)

        img_id = sample['filename'].split('.')[0]
        region_feat, region_cls, region_loc = self.get_region_feat(img_id)

        return captions, captions_w_eos, region_feat, region_cls, region_loc, \
                len(captions), str(sample['imgid'])

    def __len__(self):
        return len(self.database)


def collate_fn(batch):
    batch = list(zip(*batch))

    captions = itertools.chain(*batch[0])
    captions = pad_sequence(captions, batch_first=True, padding_value=PAD)
    captions_w_eos = itertools.chain(*batch[1])
    captions_w_eos = pad_sequence(captions_w_eos, batch_first=True, 
                                  padding_value=PAD)
    region_feat = torch.stack(batch[2], dim=0)
    region_cls = torch.stack(batch[3], dim=0)
    region_loc = torch.stack(batch[4], dim=0)

    return captions, captions_w_eos, region_feat, region_cls, region_loc, \
           batch[5], batch[6]
