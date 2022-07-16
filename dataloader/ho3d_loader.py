import os
import os.path as osp
import cv2
import pickle
import random
import numpy as np

from albumentations.pytorch import ToTensorV2
from dataloader.loader import Loader
from dataloader.transformation import get_augmentation
from helpers.util import uvd2xyz, xyz2uvd

JOINT = np.array([0,1,3,5,   6,7,9,11,  12,13,15,17,  18,19,21,23,  24,25,27,28,  32,30,31])
EVAL = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 17, 18, 21, 22, 20])
train_dirs = ['ABF10', 'ABF11', 'ABF13', 'ABF14', 'BB10', 'BB12', 'BB14', 'GPMF10', 'GPMF11', 'GPMF13', 'GPMF14', 'GSF10', 'GSF11', 'GSF12', 'GSF13', 'GSF14', 'MC5', 'MC6', 'MDF11', 'MDF12', 'MDF14', 'ND2', 'SB12', 'SB14', 'SM2', 'SM4', 'SM5', 'SMu40', 'SMu42', 'SS1', 'SS3', 'ShSu10', 'ShSu12', 'SiBF10', 'SiBF11', 'SiBF12', 'SiBF13', 'SiBF14', 'SiS1']
val_dirs = ['BB11', 'GPMF12', 'MDF10', 'MDF13', 'SB10', 'SMu41', 'SS2', 'ShSu13']
test_dirs = ['ABF12', 'BB13', 'MC1', 'MC2', 'MC4', 'SM3', 'SMu1', 'ShSu14']

class HO3D(Loader):
    
    def __init__(self, root, phase, modality="rgb", img_size=128, cube=300, jt_num=21, cwd="./", aug_type="none"):
        super(HO3D, self).__init__(root, phase, img_size, 'ho3d')
        self.name = 'ho3d'
        self.root = root
        self.phase = phase
        self.modality = modality
        self.img_size = img_size
        self.jt_num = jt_num
        self.cwd =cwd
        random.seed(0)
        assert self.modality in ["rgb"]
        
        self.dsize = np.asarray([img_size, img_size])

        self.cube = np.asarray([cube, cube, cube])
        self.dsize_crop = np.asarray([int(img_size * 1.5), int(img_size * 1.5)])
        
        self.data = self._make_dataset()
        self.flip = -1  # flip y axis when doing xyz <-> uvd transformation

        self.to_tensor = ToTensorV2()
        # assert aug_type in ["none", "weak", "strong"]
        self.transform = get_augmentation(self.phase, self.modality, aug_type, self.img_size, prob=0.3, prob2=0.25, prob3=0.5)
        
        self.origin_2d_dummy = np.asarray([[0, 0]])
        self.cube_2d_dummy = np.asarray([[self.cube[0], self.cube[1]]])
        self.dist = np.sqrt(((self.cube_2d_dummy[0] - self.origin_2d_dummy[0])**2).sum())
        self.reorder_idx = np.array([0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20])

        print(f"loading dataset: [phase: {self.phase:<5}][# images: {len(self.data):>8,}]")

    def _make_dataset(self):
        assert self.phase in ['train', 'val', 'test']
        data = []
        if self.phase == "train":
            for train_dir in train_dirs:
                _file = osp.join(self.cwd, f"data/ho3d/center_{train_dir}.txt")
                assert os.path.exists(_file)
                with open(_file) as f:
                    lines = f.readlines()
                    # img file, center x, center y, center z, cube
                    data.extend([line.strip().split(" ") for line in lines if float(line.split(" ")[3])!=0])
        elif self.phase == "val":
            for val_dir in val_dirs:
                _file = osp.join(self.cwd, f"data/ho3d/center_{val_dir}.txt")
                assert os.path.exists(_file)
                with open(_file) as f:
                    lines = f.readlines()
                    # img file, center x, center y, center z, cube
                    data.extend([line.strip().split(" ") for line in lines if float(line.split(" ")[3])!=0])
            data = sorted(data)
        elif self.phase == "test":
            for test_dir in test_dirs:
                _file = osp.join(self.cwd, f"data/ho3d/center_{test_dir}.txt")
                assert os.path.exists(_file)
                with open(_file) as f:
                    lines = f.readlines()
                    # img file, center x, center y, center z, cube
                    data.extend([line.strip().split(" ") for line in lines if float(line.split(" ")[3])!=0])
            data = sorted(data)
        img_path = osp.join(self.root, data[0][0])
        assert os.path.exists(img_path)
        img = cv2.imread(img_path)
        self.h, self.w = img.shape[:2]

        return data

    def load_pickle_data(self, f_name):
        """ Loads the pickle data """
        if not os.path.exists(f_name):
            raise Exception('Unable to find annotations picle file at %s. Aborting.'%(f_name))
        with open(f_name, 'rb') as f:
            try:
                pickle_data = pickle.load(f, encoding='latin1')
            except:
                pickle_data = pickle.load(f)

        return pickle_data

    def img_reader(self, img_path):
        img_path = osp.join(self.root, img_path)
        if self.modality == "rgb":
            img = cv2.imread(img_path)
        mask = cv2.imread(img_path.replace("rgb", "hand_mask").replace("jpg", "png"), 0)
        mask = cv2.resize(mask, img.shape[:2][::-1], interpolation=cv2.INTER_NEAREST) / 255.
        return img, mask

    def label_reader(self, img_path):
        img_path = osp.join(self.root, img_path)
        labelname = img_path.replace("rgb", "meta").replace("jpg", "pkl")
        assert osp.exists(labelname)
        # dict_keys(['objRot', 'objTrans', 'handBoundingBox', 'handJoints3D', 'objCorners3DRest', 'objCorners3D', 'camMat', 'objName', 'objLabel'])
        annot = self.load_pickle_data(labelname)
        assert annot['camMat'] is not None
        fx = annot['camMat'][0, 0]
        fy = annot['camMat'][1, 1]
        cx = annot['camMat'][0, 2]
        cy = annot['camMat'][1, 2]
        params = (fx, fy, cx, cy)

        return params, annot["handJoints3D"][self.reorder_idx] # (21, 3)

    def __getitem__(self, index):
        img, hand_mask = self.img_reader(self.data[index][0])  # data
        img_name = "_".join(self.data[index][0].split("/"))
        paras, jt_xyz = self.label_reader(self.data[index][0]) # labels_xyz
        jt_xyz[:, :2] = jt_xyz[:, :2] * 1000 # change z to positive
        jt_xyz[:, 2] = jt_xyz[:, 2] * -1000 # change z to positive

        # calc center
        center_xyz = 1000. * np.asarray([float(self.data[index][1]), -1*float(self.data[index][2]), float(self.data[index][3])], dtype=np.float32)  # center_xyz
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