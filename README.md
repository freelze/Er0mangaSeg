# Er0manga censorship segmentation

This repo contains training and testing code for the manga censorship segmentation and is based on OpenMMLab. 

Pretrained models: `To be uploaded`

## Quick start:

### Step 1: Download the pretrained model:

Download pretrained ConvNeXt model from `https://mega.nz/file/NNQTgR4Q#MuqoCZACOc9pBZ5BzafszLqa0MEnI65KJx4PXqgjV-k` and put it into the `pretrained` directory

### Step 2: Run the demo:

Run the model on the demo image located in `demo_images/1.png`:

![demo](./demo_images/1.png)

```
PYTHONPATH=. python demo/image_demo.py ./demo_images configs/convnext/convnext_h.py ./pretrained/convnext_1024_iter_400.pth --out-dir ./output_debug --mask-dir ./output_mask
```

The debug output should look like this:

![debug](./output_debug/1.png)

## Usage:

### Test the model:

```
PYTHONPATH=. python demo/image_demo.py <input image directory> configs/convnext/convnext_h.py <.pth checkpoint> --out-dir <output debug directory> --mask-dir <output mask directory>
```

With Test Time Augmentations (TTA):

```
PYTHONPATH=. python demo/image_demo_tta.py <input image directory> configs/convnext/convnext_h.py <.pth checkpoint> --mask-dir <output mask directory>
```

You can try your own TTAs that might better suit the type of manga censorship you are dealing with.


### Train the model:

I had issues with convergence, so I had to split the training in two steps:

#### Step 1 - train the model on 512x512 resolution: 

```
PYTHONPATH=. python tools/train.py configs/convnext/convnext_h_512_pretrain.py --cfg-options train_dataloader.dataset.data_root=<path to the dataset directory>
```

#### Step 2 - finetune the trained model on 1024x1024 resolution: 

```
PYTHONPATH=. python tools/train.py configs/convnext/convnext_h.py --cfg-options train_dataloader.dataset.data_root=<path to the dataset directory>
```


You should use `tools/dist_train.sh` for multi-gpu training.


## Dataset format:

The dataset is must consist of already uncensored manga pages (the pages with nudity that do not contain any censorship bars), and must be labelled with bounding boxes of the regions which probably would-be censored by the human. These bounding boxes will be augmented during training and censorship will be applied.

```
$ ls /data/data_all
annot_train.txt train
```

`train` - contains image files

`annot_train.txt` - contains lines in the format `img_name OK <bboxes>`.

Example: `img00000000_00000008.jpg OK 1279 830 1583 1145 1329 144 1436 251`
