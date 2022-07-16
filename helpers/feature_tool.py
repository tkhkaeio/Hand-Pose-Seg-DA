import numpy as np
import torch

# generate dense offsets feature 
class FeatureModule():

    def __init__(self, feat_size=64, modality="rgb", jt_num=21, kernal_size=1, heatmap_std=4, is_2D=False):
        self.feat_size = feat_size
        print("heatmap size:", self.feat_size)
        self.modality = modality
        self.jt_num = jt_num
        self.kernel_size = kernal_size
        self.is_2D = is_2D
        self.sigma = heatmap_std #self.feat_size / 64 * 4 #rgb
        size = 6 * self.sigma + 3
        x = torch.tensor(np.arange(0, size, 1, float),dtype=torch.float32)
        y = x.unsqueeze(1)
        x0, y0 = 3 * self.sigma + 1, 3 * self.sigma + 1
        self.g = torch.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))
    
    def joint2offset(self, jt_uvd, jt_uvbgr, img):
        if self.modality == "rgb" and self.is_2D:
            return self._joint2offset_rgb_2d(jt_uvbgr, img)
        else:
            raise NotImplementedError()

    def _joint2offset_rgb_2d(self, jt_uvd, img):
        batch_size = img.size(0)
        hms = torch.zeros((batch_size, self.jt_num, self.feat_size, self.feat_size), dtype = torch.float32)
        for b_idx in range(batch_size):
            keypoints = (jt_uvd[b_idx, :, :2]  + 1) * self.feat_size / 2
            for idx, pt in enumerate(keypoints):
                if pt[0] > 0: 
                    x, y = int(pt[0]), int(pt[1])
                    if x<0 or y<0 or x>=self.feat_size or y>=self.feat_size:
                        # print(x, y)
                        continue
                    ul = int(x - 3*self.sigma - 1), int(y - 3*self.sigma - 1)
                    br = int(x + 3*self.sigma + 2), int(y + 3*self.sigma + 2)

                    c,d = max(0, -ul[0]), min(br[0], self.feat_size) - ul[0]
                    a,b = max(0, -ul[1]), min(br[1], self.feat_size) - ul[1]

                    cc,dd = max(0, ul[0]), min(br[0], self.feat_size)
                    aa,bb = max(0, ul[1]), min(br[1], self.feat_size)
                    hms[b_idx, idx, aa:bb,cc:dd] = torch.maximum(hms[b_idx, idx, aa:bb,cc:dd], self.g[a:b,c:d])
        
        return hms.to(img.device)