import glob
import os
import os.path as osp
import cv2
import random
root = '/your data path/HO3D/HO3D_v3'
files = glob.glob(osp.join(root, 'train/*/seg/*.png'), recursive=True)
random.shuffle(files)
print(len(files), files[0])

for _file in files:
    _dir = "/".join(_file.split("/")[:-1]).replace("seg", "hand_mask")
    outfile = os.path.join(_dir, _file.split("/")[-1])
    if os.path.exists(outfile):
        continue

    os.makedirs(_dir, exist_ok=True)

    seg = cv2.imread(_file)
    hand_mask = seg[:, :, 0]
    cv2.imwrite(outfile, hand_mask)
