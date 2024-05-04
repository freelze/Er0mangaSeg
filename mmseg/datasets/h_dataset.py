# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .h_base_dataset import HBaseDataset


@DATASETS.register_module()
class HDataset(HBaseDataset):
    """Cityscapes dataset.

    The ``img_suffix`` is fixed to '_leftImg8bit.png' and ``seg_map_suffix`` is
    fixed to '_gtFine_labelTrainIds.png' for Cityscapes dataset.
    """
    METAINFO = dict(
        classes=('bg', 'bars',),
        palette=[[128, 0, 0], [0, 255, 0]])

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='no_label.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)
