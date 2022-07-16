# Dataset

## Download datasets
We support for DexYCB dataset and HO3D (v3) dataset. \
Download these datasets from the websites of [DexYCB](https://dex-ycb.github.io/) and [HO3D (version 3)](https://www.tugraz.at/institute/icg/research/team-lepetit/research-projects/hand-object-3d-pose-annotation/). \
Please specify your path of these datasets in `data_root` in your config file.

## Annotation generation
The data files containing file names and hand center positions are stored in `data`. \
To get hand mask labels, run `preprocess/create_hand_masks_dexycb.py` and `preprocess/create_hand_masks_ho3d.py` with your `data_root` path.