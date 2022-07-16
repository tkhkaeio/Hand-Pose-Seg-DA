# Hand-Pose-Seg-DA

This repository contains the code of our consistency training-based adaptation method for 2D hand pose estimation and hand segmentation.
## Requirements
Python 3.8 \
PyTorch 1.10.2
```
pip install -r requirements.txt
```

Please check [DATASET.md](./docs/DATASET.md) for setting up datasets. \
The training arguments are stored in `config/exp` and set your root path.


## Supervised learning
Run `scripts/train.sh` for supervised learning on dexycb. \
When you resume training from model path X and epoch Y, run `scripts/train.sh X Y`.


## Adaptation
Set your trained model used for initialization in `config/exp/config_dexycb_to_ho3d_gac.yaml`.
```
pretrain:
  load_joint_model_rgb: /set your trained model/
```

For self-training of geometric augmentation consistency (GAC),
1. train a pose branch only
```
scripts/adapt_d2h_gac.sh gac_freeze_mask
```
2. train both branches
Set the model path saved in the 1st round training to `MODEL_PATH`.
```
scripts/adapt_d2h_gac.sh gac ${MODEL_PATH}
```

## Evaluation
To evaluate a model with `MODEL_PATH`, run
```
scripts/eval.sh ${MODEL_PATH}
```