from torch.utils.data import Dataset 
import numpy as np
import cv2

class Loader(Dataset):

    def __init__(self, root, phase, img_size, dataset_name):
        assert phase in ['train', 'val', 'test']
        self.seed = np.random.RandomState(23455)
        self.root = root 
        self.phase = phase
        self.img_size = img_size
        self.dataset_name = dataset_name
        self.h = None
        self.w = None
        # randomly choose one of the augment options
        # self.aug_ops = ['trans', 'scale', 'rot', None]

    def crop_with_mask(self, img, mask, center, csize, img_size, paras):
        '''
        Crop hand region out of images, scales inverse to the distance of hand to camers
        :param center: center of mass, in image coordinates (u,v,d), d in mm
        :param csize: cube size, 3D crop volume in mm
        :param img_size: image size, resolution of cropped image, (w,h)
        :return: cropped hand image, transformation matrix for joints, center of mass in image coordinates
        '''
        assert len(csize) == 3
        assert len(img_size) == 2

        # calculate boundaries according to cube size and center
        # crop hand out of original image
        ustart, uend, vstart, vend, zstart, zend = self.center2bounds(center, csize, paras)
        img_cropped = self.bounds2crop(img, ustart, uend, vstart, vend, zstart, zend, thresh_z=(self.modality=="depth"))
        mask_cropped = self.bounds2crop(mask, ustart, uend, vstart, vend, zstart, zend, thresh_z=False)

        # resize image to same resolution
        w, h = (uend - ustart), (vend - vstart)
        # scale the longer side to corresponding img_size
        scale = min(img_size[0] / w, img_size[1] / h)
        size = (int(w * scale), int(h * scale))
        if len(img.shape) == 3:  # rgb
            # print(ustart, uend, vstart, vend, zstart, zend)
            # print(img_cropped.shape, size)
            img_cropped = cv2.resize(img_cropped, size, interpolation=cv2.INTER_CUBIC)
        else: # depth
            img_cropped = cv2.resize(img_cropped, size, interpolation=cv2.INTER_NEAREST)
        mask_cropped = cv2.resize(mask_cropped, size, interpolation=cv2.INTER_NEAREST)

        # pad another side to corresponding img_size
        mask_res = np.zeros(img_size, dtype = np.float32)
        ustart, vstart = (img_size - size) / 2.
        uend, vend = ustart + size[0], vstart + size[1]

        if len(img.shape)==3:
            img_res = cv2.copyMakeBorder(img_cropped, int(vstart), int(img_size[1] - vend + 0.5), int(ustart), int(img_size[0] - uend + 0.5), cv2.BORDER_REFLECT_101)
            # img_res = np.ones([img_size[0], img_size[1], 3], dtype = np.float32) * img.mean()
            # img_res[int(vstart):int(vend), int(ustart):int(uend), :] = img_cropped
        else:
            img_res = np.zeros(img_size, dtype = np.float32)
            img_res[int(vstart):int(vend), int(ustart):int(uend)] = img_cropped
        mask_res[int(vstart):int(vend), int(ustart):int(uend)] = mask_cropped

        transmat = self.center2transmat(center, csize, img_size, paras)

        return img_res, mask_res, transmat

    def scale_cube(self, origin_2d_dummy, cube_2d_dummy):
        dist_after = np.sqrt(((cube_2d_dummy - origin_2d_dummy)**2).sum())
        scale = dist_after / self.dist
        return self.cube.astype(np.float32) * scale

    def center2bounds(self, center, csize, paras):

        ustart, vstart = center[:2] - (csize[:2] / 2.) / center[2] * paras[:2] + 0.5
        uend, vend= center[:2] + (csize[:2] / 2.) / center[2] * paras[:2] + 0.5
        zstart = center[2] - csize[2] / 2.
        zend = center[2] + csize[2] / 2.
        
        # return int(ustart), int(uend), int(vstart), int(vend), zstart, zend
        return max(0, int(ustart)), min(int(uend), self.w), max(0, int(vstart)), min(int(vend), self.h), zstart, zend
            
    def bounds2crop(self, img, ustart, uend, vstart, vend, zstart, zend, thresh_z=True, bg=0):
        '''
        Use boundaries to crop hand out of original image.
        :return: cropped image
        '''
        h, w = img.shape[:2]
        bbox = [max(vstart,0), min(vend,h), max(ustart,0), min(uend,w)]
        img = img[bbox[0]:bbox[1], bbox[2]:bbox[3]]
        # add pixels that are out of the image in order to keep aspect ratio
        if len(img.shape)==3: 
            img = np.pad(img, ((abs(vstart)-bbox[0], abs(vend)-bbox[1]),(abs(ustart)-bbox[2], abs(uend)-bbox[3]),(0, 0)), mode='constant', constant_values=bg)
        else:
            img = np.pad(img, ((abs(vstart)-bbox[0], abs(vend)-bbox[1]),(abs(ustart)-bbox[2], abs(uend)-bbox[3])), mode='constant', constant_values=bg)

        if thresh_z:
            mask1 = np.logical_and(img < zstart, img != 0)
            mask2 = np.logical_and(img > zend, img != 0)
            img[mask1] = zstart
            img[mask2] = 0 

        return img 

    def center2transmat(self, center, csize, img_size, paras):
        '''
        Calculate affine transform matrix for scale and translate from crop.
        :param img_size: organized as (w,h), cv2 img.shape (h,w,c)
        '''
        assert len(csize) == 3
        assert len(img_size) == 2

        # calculate boundaries according to cube size and center
        # crop hand out of original image
        ustart, uend, vstart, vend, _, _ = self.center2bounds(center, csize, paras)

        trans1 = np.eye(3)
        trans1[0][2] = -ustart
        trans1[1][2] = -vstart

        w = (uend - ustart)
        h = (vend - vstart)
        # scale the longer side to corresponding img_size
        scale = min(img_size[0] / w, img_size[1] / h)
        size = (int(w * scale), int(h * scale))

        scale *= np.eye(3)
        scale[2][2] = 1

        # pad another side to corresponding img_size
        trans2 = np.eye(3)
        trans2[0][2] = int(np.floor(img_size[0] / 2. - size[0] / 2.))
        trans2[1][2] = int(np.floor(img_size[1] / 2. - size[1] / 2.))

        return np.dot(trans2, np.dot(scale, trans1)).astype(np.float32)

    def transform_jt_uvd(self, jt_uvd, M):
        pts_trans = np.hstack([jt_uvd[:,:2], np.ones((jt_uvd.shape[0], 1))])

        pts_trans = np.dot(M, pts_trans.T).T
        pts_trans[:, :2] /= pts_trans[:, 2:]

        jt_uvd = np.hstack([pts_trans[:, :2], jt_uvd[:, 2:]]).astype(np.float32)

        return jt_uvd

