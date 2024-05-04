# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
import os.path as osp
from typing import Callable, Dict, List, Optional, Sequence, Union

import mmengine
import mmengine.fileio as fileio
import numpy as np
from mmengine.dataset import BaseDataset, Compose
from mmseg.datasets import BaseSegDataset

from mmseg.registry import DATASETS


@DATASETS.register_module()
class HBaseDataset(BaseSegDataset):

    def load_data_list(self) -> List[dict]:
        """Load annotation from directory or annotation file.

        Returns:
            list[dict]: All data info of dataset.
        """
        data_list = []
        img_dir = self.data_prefix.get('img_path', None)
        ann_dir = self.data_prefix.get('seg_map_path', None)

        def read_txt(txt_name):
            d = dict()
            with open(txt_name, 'r') as f:
                for line in f:
                    name, status, *boxes = line.split()
                    boxes = [int(x) for x in boxes]
                    boxes = np.array(boxes).reshape(-1, 4)
                    d[name] = {'status': status, 'boxes': boxes}
            return d

        _suffix_len = len(self.img_suffix)
        ann_f = os.path.join(os.path.dirname(img_dir), 'annot_'+os.path.basename(img_dir))+'.txt'
        print(img_dir, self.img_suffix, ann_f)
        d = read_txt(ann_f)

        for img in fileio.list_dir_or_file(
                dir_path=img_dir,
                list_dir=False,
                suffix=self.img_suffix,
                recursive=True,
                backend_args=self.backend_args):

            if not img in d:
                continue
            if d[img]['status'] != 'OK':
                continue
            if d[img]['boxes'].shape[0] == 0:
                continue

            data_info = dict(img_path=osp.join(img_dir, img))
            if ann_dir is not None:
                seg_map = img[:-_suffix_len] + self.seg_map_suffix
                data_info['seg_map_path'] = osp.join(ann_dir, seg_map)
            data_info['label_map'] = self.label_map
            data_info['reduce_zero_label'] = self.reduce_zero_label
            data_info['seg_fields'] = []
            data_info['boxes'] = d[img]['boxes']
            data_list.append(data_info)
        data_list = sorted(data_list, key=lambda x: x['img_path'])
        print(f'Training on {len(data_list)} images!')

        return data_list


