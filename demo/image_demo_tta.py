# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
import os
import torch
import numpy as np
import cv2
from copy import deepcopy

from mmengine.model import revert_sync_batchnorm

from mmseg.apis import inference_model, init_model, show_result_pyplot
from mmseg.apis.utils import _preprare_data


def inference_tta(model, img):

    def infer(img, model, aug=None):

        if aug:
            tplo = deepcopy(model.cfg.test_pipeline)
            tpl = deepcopy(tplo)
            for v, e in enumerate(aug):
                tpl.insert(1+v, e)
            model.cfg.test_pipeline = tpl

        data, is_batch = _preprare_data(img, model)
        with torch.no_grad():
            results = model.test_step(data)
        if aug:
            model.cfg.test_pipeline = tplo

        return results if is_batch else results[0]

    augs = [
                [],


                #[
                #    {"type":'Manyfilter', 'median': True},
                #],


                [
                    {'type':'RandomFlip', 'prob': 1.0, 'direction':'horizontal'},
                    {"type":'Manyfilter', 'gamma': 0.3},
                ],


                [
                    {'type':'RandomFlip', 'prob': 1.0, 'direction':'vertical'},
                    {"type":'Manyfilter', 'gamma': 0.5},
                    {"type":'Manyfilter', 'median': True},
                ],

    ]

    total = None
    for aug in augs:
        result = infer(img, model, aug)
        if total is None:
            total = result.seg_logits_h
        else:
            total += result.seg_logits_h

    total = total/len(augs)
    i_seg_pred = np.zeros(shape=(total.shape[1], total.shape[2]), dtype=np.uint8)

    p_seg = torch.nn.functional.softmax(total, dim=0).cpu().numpy()
    THRESH = 0.4
    idx = p_seg[1, :, :] > THRESH
    i_seg_pred[idx] = 1
    i_seg_pred[~idx] = 0

    raw_mask = (p_seg[1, :, :] * 255).astype(np.uint8)

    return i_seg_pred*255, raw_mask


def main():
    parser = ArgumentParser()
    parser.add_argument('in_dir', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--mask-dir', default=None, help='Path to out mask file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    parser.add_argument(
        '--with-labels',
        action='store_true',
        default=False,
        help='Whether to display the class labels.')
    parser.add_argument(
        '--title', default='result', help='The image identifier.')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)
    if args.device == 'cpu':
        model = revert_sync_batchnorm(model)
    # test a single image
    
    files = os.listdir(args.in_dir)
    files = [f for f in files if f.endswith(('.png', ))]
    for img in files:
        in_file = os.path.join(args.in_dir, img)
        print(in_file)

        out_mask, raw_mask = inference_tta(model, in_file)

        #out_mask = result.pred_sem_seg.cpu().data.cpu().numpy().astype(np.uint8)[0]*255
        out_mask_fname = os.path.join(args.mask_dir, img)
        cv2.imwrite(out_mask_fname, out_mask)


if __name__ == '__main__':
    main()

