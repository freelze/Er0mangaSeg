# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
import os
import numpy as np
import cv2

from mmengine.model import revert_sync_batchnorm

from mmseg.apis import inference_model, init_model, show_result_pyplot


def main():
    parser = ArgumentParser()
    parser.add_argument('in_dir', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--mask-dir', default=None, help='Path to out mask file')
    parser.add_argument('--out-dir', default=None, help='Path to output file')
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
    files = [f for f in files if f.endswith(('.png', 'jpg', 'jpeg'))]
    for img in files:
        in_file = os.path.join(args.in_dir, img)
        print(in_file)
        result = inference_model(model, in_file)
        out_mask = result.pred_sem_seg.cpu().data.cpu().numpy().astype(np.uint8)[0]*255
        out_mask_fname = os.path.join(args.mask_dir, img)
        cv2.imwrite(out_mask_fname, out_mask)
        # show the results
        show_result_pyplot(
            model,
            in_file,
            result,
            title=args.title,
            opacity=args.opacity,
            with_labels=args.with_labels,
            draw_gt=False,
            show=False if args.out_dir is not None else True,
            out_file=os.path.join(args.out_dir, img))


if __name__ == '__main__':
    main()
