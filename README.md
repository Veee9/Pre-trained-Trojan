# Pre-trained-Trojan

## Introduction

This repository is the official PyTorch implemetation of paper "Pre-trained Trojan Attacks".

## Install

### Requirements

* Python >= 3.6

* PyTorch >= 1.5

```shell
pip install -r requirements.txt
```

#### Data Preparation

In order to train a trigger generation network, it is necessary to  prepare a training dataset. We utilize images from the COCO training set for this purpose.

The data should look like this:

```
dataset_root
|-- img1
|-- img2
|-- img3
|……
```

#### Style Images Preparation

Prepare the style images for the target label and place them in a folder, then move the folder to the "style_imgs" directory.

The images should look like this:

```
style_images_root
|-- img1
|-- img2
|-- img3
|……
```

## Usage

### Generate trigger

Train a trigger generation network:

```shell
cd gen_trigger

python gen.py train --style_image_dir style_imgs/banana_32 \
    --style_image_num 32 \
    --dataset dataset_root \
    --gpu 0 --visualize 1
```

Generate a trigger from a model:

```shell
python gen.py transfer \
    --model-path xxx/xxx.model \
    --source content_imgs/fl.jpg \
    --output fl_st_banana.jpg \
    --gpu 0
```

### Generate poisoned images

Place a trigger image into the "source_image" folder. Make sure that its file extension is ".jpg".

```shell
cd gen_backdoor_dataset

python make_dataset.py --name hk_st_banana \
    --attack_label 954 \
    --gen_train_num 1300 \
    --gen_val_num 50
```

