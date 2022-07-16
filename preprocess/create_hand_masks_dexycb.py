import os
import os.path as osp
import numpy as np
from glob import glob as glob
import cv2
import random

root = '/your data path/DexYCB'
files = glob(osp.join(root, '20*/**/labels*.npz'), recursive=True)
print(len(files), files[0])
random.shuffle(files)
for _file in files:
    outpath = _file.replace(".npz", ".png").replace("labels", "hand_mask")
    if os.path.exists(outpath):
        continue
    with np.load(_file) as data:
        seg = data["seg"]
        hand_seg = np.where(seg != 255, 0, 255)
        cv2.imwrite(outpath, hand_seg)
print("done")
