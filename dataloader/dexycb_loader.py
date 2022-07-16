import os
import os.path as osp
import cv2
import yaml
import random
import numpy as np
from glob import glob

from albumentations.pytorch import ToTensorV2
from dataloader.loader import Loader
from dataloader.transformation import get_augmentation
from helpers.util import uvd2xyz, xyz2uvd

JOINT = np.array([0,1,3,5,   6,7,9,11,  12,13,15,17,  18,19,21,23,  24,25,27,28,  32,30,31])
EVAL = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 17, 18, 21, 22, 20])
VIEWS = {0: "836212060125", 1: "839512060362", 2: "840412060917", 3: "841412060263", 4: "932122060857", 5: "932122060861", 6: "932122061900", 7: "932122062010"}
VIEWS_INV = {v: k for k, v in VIEWS.items()}
class DexYCB(Loader):
    
    def __init__(self, root, phase, modality="rgb", img_size=128, cube=300, jt_num=21, cwd="./", aug_type="none"):
        super(DexYCB, self).__init__(root, phase, img_size, 'dexycb')
        self.name = 'dexycb'
        self.root = root
        self.phase = phase
        self.modality = modality
        self.img_size = img_size
        self.jt_num = jt_num
        self.cwd = cwd
        random.seed(0)
        assert self.modality in ["rgb"]

        self.paras_dict = self._make_camera_params()
        self.dsize = np.asarray([img_size, img_size])

        self.cube = np.asarray([cube, cube, cube])
        self.dsize_crop = np.asarray([int(img_size * 1.5), int(img_size * 1.5)])

        self.data = self._make_dataset()
        self.flip = 1  # -1  # flip y axis when doing xyz <-> uvd transformation

        self.to_tensor = ToTensorV2()
        assert aug_type in ["none", "weak", "strong"]
        self.transform = get_augmentation(self.phase, self.modality, aug_type, self.img_size, prob=0.3, prob2=0.25, prob3=0.5)
        self.origin_2d_dummy = np.asarray([[0, 0]])
        self.cube_2d_dummy = np.asarray([[self.cube[0], self.cube[1]]])
        self.dist = np.sqrt(((self.cube_2d_dummy[0] - self.origin_2d_dummy[0])**2).sum())
        print(f"loading dataset: [phase: {self.phase:<5}][# images: {len(self.data):>8,}]")
    

    def _make_camera_params(self):
        self.paras_dict = {}
        for key, view_name in VIEWS.items():
            intrname = osp.join(self.root, f"calibration/intrinsics/{view_name}_640x480.yml")
            assert osp.exists(intrname)
            with open(intrname, 'r') as f:
                intr = yaml.load(f, Loader=yaml.FullLoader)
                if self.modality=="rgb":
                    fx = intr['color']['fx']
                    fy = intr['color']['fy']
                    cx = intr['color']['ppx']
                    cy = intr['color']['ppy']
                    _paras = (fx, fy, cx, cy)
                else:
                    raise NotImplementedError()
            self.paras_dict[key] = _paras
        return self.paras_dict

    def _make_dataset(self):
        assert self.phase in ['train', 'val', 'test']
        data = []
        if self.phase == "train":
            for sub_id in range(1, 7): #[1-6]
                files = glob(osp.join(self.cwd, f"data/dexycb/center_{self.modality}_sub{sub_id:02d}.txt"), recursive=True)
                assert len(files) == 1
                with open(files[0]) as f:
                    lines = f.readlines()
                    # img file, center x, center y, center z, cube
                    data.extend([line.strip().split(" ") for line in lines if float(line.split(" ")[3])!=0])
        elif self.phase == "val": #[7, 8]
            for sub_id in range(7, 9):
                files = glob(osp.join(self.cwd, f"data/dexycb/center_{self.modality}_sub{sub_id:02d}.txt"), recursive=True)
                assert len(files) == 1
                with open(files[0]) as f:
                    lines = f.readlines()
                    # img file, center x, center y, center z, cube
                    data.extend([line.strip().split(" ") for line in lines if float(line.split(" ")[3])!=0])
            data = sorted(data)
        elif self.phase == "test": #[9, 10]
            for sub_id in range(9, 11):
                files = glob(osp.join(self.cwd, f"data/dexycb/center_{self.modality}_sub{sub_id:02d}.txt"), recursive=True)
                assert len(files) == 1
                with open(files[0]) as f:
                    lines = f.readlines()
                    # img file, center x, center y, center z, cube
                    data.extend([line.strip().split(" ") for line in lines if float(line.split(" ")[3])!=0])
            data = sorted(data)
        img_path = osp.join(self.root, data[0][0])
        assert os.path.exists(img_path)
        img = cv2.imread(img_path)
        self.h, self.w = img.shape[:2]

        return data

    def img_reader(self, img_path):
        img_path = osp.join(self.root, img_path)
        if self.modality == "rgb":
            img = cv2.imread(img_path)
        view_name = img_path.split("/")[-2]
        view_id = VIEWS_INV[view_name]
        return img, view_id

    def label_reader(self, img_path):
        img_path = osp.join(self.root, img_path)
        if self.modality == "rgb":
            labelname = img_path.replace("color", "labels").replace("jpg", "npz")
        assert osp.exists(labelname)
        with np.load(labelname) as labels:
            seg = labels["seg"]                                   # (480, 640)
            hand_mask = np.where(seg!=255, 0, 1.)
            joint_3d = labels["joint_3d"].reshape(self.jt_num, 3) # (21, 3)
            joint_2d = labels["joint_2d"].reshape(self.jt_num, 2) # (21, 2)

        return {"hand_mask": hand_mask, "joint_3d": joint_3d, "joint_2d": joint_2d}
        
    def __getitem__(self, index):
        img, view_id = self.img_reader(self.data[index][0])  # data
        img_name = "_".join(self.data[index][0].split("/"))
        paras = self.paras_dict[view_id]
        label = self.label_reader(self.data[index][0])      # labels_xyz
        hand_mask = label["hand_mask"]
        jt_xyz = label["joint_3d"] * 1000.
            
        # calc center
        center_xyz = 1000. * np.asarray([float(self.data[index][1]), float(self.data[index][2]), float(self.data[index][3])], dtype=np.float32)  # center_xyz
        center_uvd = xyz2uvd(center_xyz, paras, self.flip)
        # image cropping & keypoint transformation
        img, mask, M = self.crop_with_mask(img, hand_mask, center_uvd, self.cube, self.dsize_crop, paras)
        jt_uvd = xyz2uvd(jt_xyz, paras, self.flip)
        jt_uvd = self.transform_jt_uvd(jt_uvd, M)
        center_uvd = self.transform_jt_uvd(np.asarray([center_uvd]), M)[0]

        keypoints = np.concatenate((jt_uvd[:, :2], np.expand_dims(center_uvd[:2], axis=0), self.origin_2d_dummy, self.cube_2d_dummy), axis=0)
        # data augmentation
        if self.modality == "rgb":
            transformed = self.transform(image=img, keypoints=keypoints, mask=mask)
            img, keypoints, mask = transformed['image'], np.asarray(transformed['keypoints']), transformed['mask'].unsqueeze(0)
            # calc scale factor and modify cube
            cube = self.scale_cube(keypoints[self.jt_num+1], keypoints[self.jt_num+2]) if self.phase == "train" else self.cube
        jt_uvd_2d = keypoints[:self.jt_num]
        center_uvd[:2]= keypoints[self.jt_num]

        # calc new center
        center_xyz = uvd2xyz(center_uvd, paras, self.flip)

        # register valid joints 
        jt_uvd[:, :2] = jt_uvd_2d
        jt_valid = np.zeros(self.jt_num)
        jt_valid = (jt_uvd[:, 1] > 0) * (jt_uvd[:, 0] > 0) * (jt_uvd[:, 1] < self.img_size-1) * (jt_uvd[:, 0] < self.img_size-1)
        # normalize joint keypoints
        jt_uvd[:, :2] = jt_uvd_2d / (self.img_size / 2.) - 1
        jt_uvd[:, 2] = (jt_uvd[:, 2] - center_xyz[2]) / (cube[2] / 2.0)

        # uv coordinate + rgb value
        jt_uvbgr = np.zeros((self.jt_num, 5))
        if self.modality == "rgb":
            for i, is_valid in enumerate(jt_valid):
                if is_valid:
                    jt_uvbgr[i, 2:] = img[:, int(jt_uvd[i, 1]), int(jt_uvd[i, 0])]
            jt_uvbgr[:, :2] = jt_uvd[:, :2]
        
        # world coordinate is originated in hand center
        jt_xyz -= center_xyz
        jt_xyz = jt_xyz / (cube / 2.)
        
        return {"img": img,
                "mask": mask.float(),
                "jt_xyz_gt": jt_xyz.astype(np.float32),
                "jt_uvd_gt": jt_uvd.astype(np.float32),
                "jt_uvbgr_gt": jt_uvbgr.astype(np.float32),
                "center_xyz": center_xyz.astype(np.float32),
                "M": M.astype(np.float32), 
                "jt_valid": jt_valid.astype(np.float32), 
                "cube": cube.astype(np.float32),
                "paras": np.asarray(list(paras), dtype=np.float32),
                "img_name": img_name,
                "index": index
                }

    def __len__(self):
        return len(self.data)