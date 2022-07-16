import random
import warnings
import numpy as np

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations import functional as F
from albumentations.core.transforms_interface import DualTransform
from skimage.transform._geometric import ProjectiveTransform

warnings.resetwarnings()
warnings.simplefilter('ignore', UserWarning)

def get_augmentation(phase, modality, aug_type="strong", img_size=128, prob=0.3, prob2=0.25, prob3=0.5):
    """
    Adjust augmentation parameters for hand segmentation and hand pose estimation
    """
    assert modality == "rgb"
    if phase=="train":
        if "strong" in aug_type:
            transform_list = [
                A.Affine(scale=(0.5, 1.5), p=prob, mode=4),
                A.Rotate(limit=90, p=prob),
                A.HorizontalFlip(p=prob),
                A.RandomCrop(img_size, img_size),
                A.Blur(blur_limit=[3,7], p=prob),
                A.GaussNoise(p=prob, var_limit=(10.0, 200)),
                A.Solarize(p=prob2),
                A.RandomBrightnessContrast(p=prob2, brightness_limit=1, contrast_limit=1),
                A.HueSaturationValue(p=prob2, hue_shift_limit=(-40, 40), sat_shift_limit=(-40, 40), val_shift_limit=(-40, 40)),
                A.Sharpen(p=prob2),
                CoarseDropout(max_holes=5, min_holes=1, max_height=int(img_size * 0.2), min_height=int(img_size * 0.05),
                            max_width=int(img_size * 0.2), min_width=int(img_size * 0.05), mean_fill=True, mask_fill_value=0, p=prob3),
                A.Normalize(mean=(0.5,), std=(0.5,)),
                ToTensorV2(),
            ]
        elif aug_type=="weak":
            transform_list = [
                A.Rotate(limit=45, p=prob),
                A.HorizontalFlip(p=prob),
                A.RandomCrop(img_size, img_size),
                A.Blur(blur_limit=[3,5], p=prob),
                A.RandomBrightnessContrast(p=prob2),
                A.HueSaturationValue(p=prob2),
                A.Normalize(mean=(0.5,), std=(0.5,)),
                ToTensorV2(),
            ]
        elif aug_type=="none":
            transform_list = [A.CenterCrop(img_size, img_size)]           
            transform_list.extend([
                A.Normalize(mean=(0.5,), std=(0.5,)),
                ToTensorV2(),
            ])
        else:
            raise NotImplementedError()
    else:
        transform_list = [A.CenterCrop(img_size, img_size)]
        transform_list.extend([
            A.Normalize(mean=(0.5,), std=(0.5,)),
            ToTensorV2(),
        ])
    transform = A.Compose(transform_list,
                keypoint_params=A.KeypointParams(format='xy', remove_invisible=False),)
    return transform

class TestTimeAugmentation(object):
    # default: strong augmentation
    def __init__(self, aug_type="strong", modality="rgb", img_size=128, jt_num=21, prob=0.3, prob2=0.25, prob3=0.5):
        self.modality = modality
        assert self.modality == "rgb"
        self.img_size = img_size
        self.jt_num = jt_num
        if aug_type=="strong":
            transform_list = [
                A.CenterCrop(img_size, img_size),
                A.HorizontalFlip(p=prob),
                A.Affine(rotate=(-90, 90), p=prob, mode=4),
                A.Affine(translate_percent=(0, 0.2), p=prob, mode=4),
                A.Blur(blur_limit=[3,7], p=prob),
                A.GaussNoise(p=prob, var_limit=(10.0, 200)),
                A.Solarize(p=prob2),
                A.RandomBrightnessContrast(p=prob2),
                A.HueSaturationValue(p=prob2, hue_shift_limit=(-40, 40), sat_shift_limit=(-40, 40), val_shift_limit=(-40, 40)),
                A.Sharpen(p=prob2),
                CoarseDropout(max_holes=5, min_holes=1, max_height=int(img_size * 0.2), min_height=int(img_size * 0.05),
                            max_width=int(img_size * 0.2), min_width=int(img_size * 0.05), mean_fill=True, mask_fill_value=0, p=prob3),
                A.Normalize(mean=(0.5,), std=(0.5,)),
                ToTensorV2(),
            ]
        elif aug_type=="weak":
            transform_list = [
                A.CenterCrop(img_size, img_size),
                A.HorizontalFlip(p=prob),
                A.Affine(rotate=(-45, 45), p=prob, mode=4),
                A.Affine(translate_percent=(0, 0.2), p=prob, mode=4),
                A.Blur(blur_limit=[3,5], p=prob),
                A.RandomBrightnessContrast(p=prob2),
                A.HueSaturationValue(p=prob2),
                CoarseDropout(max_holes=5, min_holes=1, max_height=int(img_size * 0.2), min_height=int(img_size * 0.05),
                    max_width=int(img_size * 0.2), min_width=int(img_size * 0.05), mean_fill=True, mask_fill_value=0, p=prob3), #TODO: check
                A.Normalize(mean=(0.5,), std=(0.5,)),
                ToTensorV2(),
            ]
        elif aug_type=="color":
            transform_list = [
                A.CenterCrop(img_size, img_size),
                A.Blur(blur_limit=[3,5], p=prob),
                A.RandomBrightnessContrast(p=prob2),
                A.HueSaturationValue(p=prob2),
                A.Normalize(mean=(0.5,), std=(0.5,)),
                ToTensorV2(),
            ]
        elif aug_type=="weak_weak":
            transform_list = [
                A.HorizontalFlip(p=prob),
                A.Affine(rotate=(-10, 10), p=prob, mode=4),
                A.Blur(blur_limit=[3,5], p=prob),
                A.Normalize(mean=(0.5,), std=(0.5,)),
                ToTensorV2(),
            ]
        self.transform = A.ReplayCompose(transform_list,
                keypoint_params=A.KeypointParams(format='xy', remove_invisible=False),)

    def deprocess(self, image, keypoints, mask):
        keypoints = keypoints.numpy()
        img_numpy = (np.transpose(image.numpy(), (1, 2, 0)) + 1 )/2
        img_numpy = np.clip(img_numpy * 255, 0, 255)
        keypoints[:, :2] = (keypoints[:, :2] + 1) * (self.img_size / 2.)
        mask_numpy = np.transpose(mask.numpy(), (1, 2, 0))
        return img_numpy, keypoints, mask_numpy
    
    def preprocess(self, image, keypoints, mask):
        keypoints[:, :2] = keypoints[:, :2] / (self.img_size / 2.) - 1
        keypoints = torch.from_numpy(keypoints)
        mask_numpy = np.transpose(mask, (2, 0, 1))
        mask = torch.from_numpy(mask_numpy)
        return image, keypoints, mask
    
    def _check_funcs(self):
        seed = 1234
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        image = torch.rand(1, 3, self.img_size, self.img_size)
        mask = torch.rand(1, 1, self.img_size, self.img_size)
        keypoints = torch.rand(1, 21, 3)/2
        print("check", np.round(mask[0, 0, 60, 60], 4))
        image_nw, keypoints_nw, mask_nw = self.__call__(image, keypoints.clone(), mask)
        keypoints_bk, mask_bk = self.inverse_apply(image_nw, keypoints_nw, mask_nw)
        keypoints, mask, keypoints_bk, mask_bk = keypoints.numpy(), mask.numpy(), keypoints_bk.numpy(), mask_bk.numpy()
        print("check", np.round(mask[0, 0, 60, 60], 4), np.round(mask_bk[0, 0, 60, 60], 4))
        assert np.round(keypoints[0, 0, 0], 3) == np.round(keypoints_bk[0, 0, 0], 3)
        assert np.round(keypoints[0, 0, 1], 3) == np.round(keypoints_bk[0, 0, 1], 3)
        assert np.round(mask[0, 0, 60, 60], 4) == np.round(mask_bk[0, 0, 60, 60], 4)
        assert np.round(mask[0, 0, 70, 70], 4) == np.round(mask_bk[0, 0, 70, 70], 4)
        print("test has passed")

    def __call__(self, image, keypoints, mask):
        # check consistency
        # self.keypoints = keypoints.clone()
        # self.mask = mask
        image_list, keypoints_list, mask_list = [], [], []
        self.replay_list = []
        try:
            assert image.size(0) == keypoints.size(0) == mask.size(0)
        except:
            print(image.size(), keypoints.size(), mask.size())
        for i in range(image.size(0)):
            _image, _keypoints, _mask = self.deprocess(image[i], keypoints[i], mask[i])
            transformed = self.transform(image=_image, keypoints=_keypoints[:, :2], mask=_mask)
            _image, _keypoints_2d, _mask = transformed['image'], np.asarray(transformed['keypoints']), np.asarray(transformed['mask'])
            _keypoints[:, :2] = _keypoints_2d
            _image, _keypoints, _mask = self.preprocess(_image, _keypoints, _mask)
            image_list.append(_image)
            keypoints_list.append(_keypoints)
            mask_list.append(_mask)
            self.replay_list.append(transformed["replay"])
        return torch.stack(image_list, 0), torch.stack(keypoints_list, 0), torch.stack(mask_list, 0)
    
    def inverse_apply(self, image, keypoints, mask):
        keypoints_list, mask_list = [], []
        for i, replay in enumerate(self.replay_list):
            # make replay to inverse functions
            nw_transform = []
            for j, transform_func in enumerate(replay['transforms']):
                if "Affine" in transform_func['__class_fullname__'] and transform_func['params'] is not None:
                    transform_func['params']["matrix"] = ProjectiveTransform(transform_func['params']["matrix"]._inv_matrix)
                    nw_transform.append(transform_func)
                elif "CenterCrop" in transform_func['__class_fullname__'] or "Blur" in transform_func['__class_fullname__'] or "CoarseDropout" in transform_func['__class_fullname__'] or "Normalize" in transform_func['__class_fullname__'] or "ToTensor" in transform_func['__class_fullname__']:
                    pass
            nw_transform = nw_transform[::-1]
            replay['transforms'] = nw_transform
            
            _image, _keypoints, _mask = self.deprocess(image[i], keypoints[i], mask[i])
            transformed = A.ReplayCompose.replay(replay, image=_image, keypoints=_keypoints[:, :2], mask=_mask)
            _image, _keypoints_2d, _mask = transformed['image'], np.asarray(transformed['keypoints']), np.asarray(transformed['mask'])
            _keypoints[:, :2] = _keypoints_2d
            _, _keypoints, _mask = self.preprocess(_image, _keypoints, _mask)
            keypoints_list.append(_keypoints)
            mask_list.append(_mask)
        keypoints, mask = torch.stack(keypoints_list, 0), torch.stack(mask_list, 0)
        # check consistency
        # try:
        #     assert np.round(self.mask[0, 0, 60, 60], 4) == np.round(mask[0, 0, 60, 60], 4)
        #     assert np.round(self.mask[0, 0, 70, 70], 4) == np.round(mask[0, 0, 70, 70], 4)
        #     assert np.round(self.keypoints[0, 0, 0], 3) == np.round(keypoints[0, 0, 0], 3)
        #     assert np.round(self.keypoints[0, 0, 1], 3) == np.round(keypoints[0, 0, 1], 3)
        # except:
        #     print(np.round(self.keypoints[0, 0], 4), np.round(keypoints[0, 0], 4))
        #     print(np.round(self.mask[0, 0, 60, 60], 4), np.round(mask[0, 0, 60, 60], 4))
        #     raise Exception()
        return keypoints, mask

class Cutout(DualTransform):
    """CoarseDropout of the square regions in the image.
    Args:
        num_holes (int): number of regions to zero out
        max_h_size (int): maximum height of the hole
        max_w_size (int): maximum width of the hole
        fill_value (int, float, list of int, list of float): value for dropped pixels.
    Targets:
        image, mask
    Image types:
        uint8, float32
    Reference:
    |  https://arxiv.org/abs/1708.04552
    |  https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
    |  https://github.com/aleju/imgaug/blob/master/imgaug/augmenters/arithmetic.py
    """

    def __init__(
        self,
        num_holes=8,
        max_h_size=8,
        max_w_size=8,
        fill_value=0,
        mean_fill=False,
        always_apply=False,
        p=0.5,
    ):
        super(Cutout, self).__init__(always_apply, p)
        self.num_holes = num_holes
        self.max_h_size = max_h_size
        self.max_w_size = max_w_size
        self.mean_fill = mean_fill
        self.fill_value = fill_value
        warnings.warn(
            "This class has been deprecated. Please use CoarseDropout",
            FutureWarning,
        )

    def apply(self, image, fill_value=0, holes=(), **params):
        if self.mean_fill:
            fill_value = image.mean()
        return F.cutout(image, holes, fill_value)

    def apply_to_keypoint(self, keypoint, **params):
        return keypoint

    def get_params_dependent_on_targets(self, params):
        img = params["image"]
        height, width = img.shape[:2]

        holes = []
        for _n in range(self.num_holes):
            y = random.randint(0, height)
            x = random.randint(0, width)

            y1 = np.clip(y - self.max_h_size // 2, 0, height)
            y2 = np.clip(y1 + self.max_h_size, 0, height)
            x1 = np.clip(x - self.max_w_size // 2, 0, width)
            x2 = np.clip(x1 + self.max_w_size, 0, width)
            holes.append((x1, y1, x2, y2))

        return {"holes": holes}

    @property
    def targets_as_params(self):
        return ["image"]

    def get_transform_init_args_names(self):
        return ("num_holes", "max_h_size", "max_w_size")



class CoarseDropout(DualTransform):
    """CoarseDropout of the rectangular regions in the image.
    Args:
        max_holes (int): Maximum number of regions to zero out.
        max_height (int, float): Maximum height of the hole.
        If float, it is calculated as a fraction of the image height.
        max_width (int, float): Maximum width of the hole.
        If float, it is calculated as a fraction of the image width.
        min_holes (int): Minimum number of regions to zero out. If `None`,
            `min_holes` is be set to `max_holes`. Default: `None`.
        min_height (int, float): Minimum height of the hole. Default: None. If `None`,
            `min_height` is set to `max_height`. Default: `None`.
            If float, it is calculated as a fraction of the image height.
        min_width (int, float): Minimum width of the hole. If `None`, `min_height` is
            set to `max_width`. Default: `None`.
            If float, it is calculated as a fraction of the image width.
        fill_value (int, float, list of int, list of float): value for dropped pixels.
        mask_fill_value (int, float, list of int, list of float): fill value for dropped pixels
            in mask. If `None` - mask is not affected. Default: `None`.
    Targets:
        image, mask
    Image types:
        uint8, float32
    Reference:
    |  https://arxiv.org/abs/1708.04552
    |  https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
    |  https://github.com/aleju/imgaug/blob/master/imgaug/augmenters/arithmetic.py
    """

    def __init__(
        self,
        max_holes=8,
        max_height=8,
        max_width=8,
        min_holes=None,
        min_height=None,
        min_width=None,
        mean_fill=False,
        max_fill=False,
        fill_value=0,
        mask_fill_value=None,
        always_apply=False,
        p=0.5,
    ):
        super(CoarseDropout, self).__init__(always_apply, p)
        self.max_holes = max_holes
        self.max_height = max_height
        self.max_width = max_width
        self.min_holes = min_holes if min_holes is not None else max_holes
        self.min_height = min_height if min_height is not None else max_height
        self.min_width = min_width if min_width is not None else max_width
        self.mean_fill = mean_fill
        self.max_fill = max_fill
        self.fill_value = fill_value
        self.mask_fill_value = mask_fill_value
        if not 0 < self.min_holes <= self.max_holes:
            raise ValueError("Invalid combination of min_holes and max_holes. Got: {}".format([min_holes, max_holes]))

        self.check_range(self.max_height)
        self.check_range(self.min_height)
        self.check_range(self.max_width)
        self.check_range(self.min_width)

        if not 0 < self.min_height <= self.max_height:
            raise ValueError(
                "Invalid combination of min_height and max_height. Got: {}".format([min_height, max_height])
            )
        if not 0 < self.min_width <= self.max_width:
            raise ValueError("Invalid combination of min_width and max_width. Got: {}".format([min_width, max_width]))

    def check_range(self, dimension):
        if isinstance(dimension, float) and not 0 <= dimension < 1.0:
            raise ValueError(
                "Invalid value {}. If using floats, the value should be in the range [0.0, 1.0)".format(dimension)
            )

    def apply(self, image, fill_value=0, holes=(), **params):
        if self.mean_fill:
            fill_value = image.mean()
        elif self.max_fill:
            fill_value = image.max() - 1
        return F.cutout(image, holes, fill_value)

    def apply_to_mask(self, image, mask_fill_value=0, holes=(), **params):
        if mask_fill_value is None:
            return image
        return F.cutout(image, holes, mask_fill_value)

    def apply_to_keypoint(self, keypoint, **params):
        return keypoint

    def get_params_dependent_on_targets(self, params):
        img = params["image"]
        height, width = img.shape[:2]

        holes = []
        for _n in range(random.randint(self.min_holes, self.max_holes)):
            if all(
                [
                    isinstance(self.min_height, int),
                    isinstance(self.min_width, int),
                    isinstance(self.max_height, int),
                    isinstance(self.max_width, int),
                ]
            ):
                hole_height = random.randint(self.min_height, self.max_height)
                hole_width = random.randint(self.min_width, self.max_width)
            elif all(
                [
                    isinstance(self.min_height, float),
                    isinstance(self.min_width, float),
                    isinstance(self.max_height, float),
                    isinstance(self.max_width, float),
                ]
            ):
                hole_height = int(height * random.uniform(self.min_height, self.max_height))
                hole_width = int(width * random.uniform(self.min_width, self.max_width))
            else:
                raise ValueError(
                    "Min width, max width, \
                    min height and max height \
                    should all either be ints or floats. \
                    Got: {} respectively".format(
                        [
                            type(self.min_width),
                            type(self.max_width),
                            type(self.min_height),
                            type(self.max_height),
                        ]
                    )
                )

            y1 = random.randint(0, height - hole_height)
            x1 = random.randint(0, width - hole_width)
            y2 = y1 + hole_height
            x2 = x1 + hole_width
            holes.append((x1, y1, x2, y2))

        return {"holes": holes}

    @property
    def targets_as_params(self):
        return ["image"]

    def get_transform_init_args_names(self):
        return (
            "max_holes",
            "max_height",
            "max_width",
            "min_holes",
            "min_height",
            "min_width",
            "fill_value",
            "mask_fill_value",
        )


class GridDropout(DualTransform):
    """GridDropout, drops out rectangular regions of an image and the corresponding mask in a grid fashion.
    Args:
        ratio (float): the ratio of the mask holes to the unit_size (same for horizontal and vertical directions).
            Must be between 0 and 1. Default: 0.5.
        unit_size_min (int): minimum size of the grid unit. Must be between 2 and the image shorter edge.
            If 'None', holes_number_x and holes_number_y are used to setup the grid. Default: `None`.
        unit_size_max (int): maximum size of the grid unit. Must be between 2 and the image shorter edge.
            If 'None', holes_number_x and holes_number_y are used to setup the grid. Default: `None`.
        holes_number_x (int): the number of grid units in x direction. Must be between 1 and image width//2.
            If 'None', grid unit width is set as image_width//10. Default: `None`.
        holes_number_y (int): the number of grid units in y direction. Must be between 1 and image height//2.
            If `None`, grid unit height is set equal to the grid unit width or image height, whatever is smaller.
        shift_x (int): offsets of the grid start in x direction from (0,0) coordinate.
            Clipped between 0 and grid unit_width - hole_width. Default: 0.
        shift_y (int): offsets of the grid start in y direction from (0,0) coordinate.
            Clipped between 0 and grid unit height - hole_height. Default: 0.
        random_offset (boolean): weather to offset the grid randomly between 0 and grid unit size - hole size
            If 'True', entered shift_x, shift_y are ignored and set randomly. Default: `False`.
        fill_value (int): value for the dropped pixels. Default = 0
        mask_fill_value (int): value for the dropped pixels in mask.
            If `None`, transformation is not applied to the mask. Default: `None`.
    Targets:
        image, mask
    Image types:
        uint8, float32
    References:
        https://arxiv.org/abs/2001.04086
    """

    def __init__(
        self,
        ratio: float = 0.5,
        unit_size_min: int = None,
        unit_size_max: int = None,
        holes_number_x: int = None,
        holes_number_y: int = None,
        shift_x: int = 0,
        shift_y: int = 0,
        random_offset: bool = False,
        mean_fill: bool = False,
        fill_value: int = 0,
        mask_fill_value: int = None,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super(GridDropout, self).__init__(always_apply, p)
        self.ratio = ratio
        self.unit_size_min = unit_size_min
        self.unit_size_max = unit_size_max
        self.holes_number_x = holes_number_x
        self.holes_number_y = holes_number_y
        self.shift_x = shift_x
        self.shift_y = shift_y
        self.random_offset = random_offset
        self.mean_fill = mean_fill
        self.fill_value = fill_value
        self.mask_fill_value = mask_fill_value
        if not 0 < self.ratio <= 1:
            raise ValueError("ratio must be between 0 and 1.")

    def apply(self, image, holes=(), **params):
        if self.mean_fill:
            return F.cutout(image, holes, image.mean())
        else:
            return F.cutout(image, holes, self.fill_value)

    def apply_to_mask(self, image, holes=(), **params):
        if self.mask_fill_value is None:
            return image

        return F.cutout(image, holes, self.mask_fill_value)

    def apply_to_keypoint(self, keypoint, **params):
        return keypoint

    def get_params_dependent_on_targets(self, params):
        img = params["image"]
        height, width = img.shape[:2]
        # set grid using unit size limits
        if self.unit_size_min and self.unit_size_max:
            if not 2 <= self.unit_size_min <= self.unit_size_max:
                raise ValueError("Max unit size should be >= min size, both at least 2 pixels.")
            if self.unit_size_max > min(height, width):
                raise ValueError("Grid size limits must be within the shortest image edge.")
            unit_width = random.randint(self.unit_size_min, self.unit_size_max + 1)
            unit_height = unit_width
        else:
            # set grid using holes numbers
            if self.holes_number_x is None:
                unit_width = max(2, width // 10)
            else:
                if not 1 <= self.holes_number_x <= width // 2:
                    raise ValueError("The hole_number_x must be between 1 and image width//2.")
                unit_width = width // self.holes_number_x
            if self.holes_number_y is None:
                unit_height = max(min(unit_width, height), 2)
            else:
                if not 1 <= self.holes_number_y <= height // 2:
                    raise ValueError("The hole_number_y must be between 1 and image height//2.")
                unit_height = height // self.holes_number_y

        hole_width = int(unit_width * self.ratio)
        hole_height = int(unit_height * self.ratio)
        # min 1 pixel and max unit length - 1
        hole_width = min(max(hole_width, 1), unit_width - 1)
        hole_height = min(max(hole_height, 1), unit_height - 1)
        # set offset of the grid
        if self.shift_x is None:
            shift_x = 0
        else:
            shift_x = min(max(0, self.shift_x), unit_width - hole_width)
        if self.shift_y is None:
            shift_y = 0
        else:
            shift_y = min(max(0, self.shift_y), unit_height - hole_height)
        if self.random_offset:
            shift_x = random.randint(0, unit_width - hole_width)
            shift_y = random.randint(0, unit_height - hole_height)
        holes = []
        for i in range(width // unit_width + 1):
            for j in range(height // unit_height + 1):
                x1 = min(shift_x + unit_width * i, width)
                y1 = min(shift_y + unit_height * j, height)
                x2 = min(x1 + hole_width, width)
                y2 = min(y1 + hole_height, height)
                holes.append((x1, y1, x2, y2))

        return {"holes": holes}

    @property
    def targets_as_params(self):
        return ["image"]

    def get_transform_init_args_names(self):
        return (
            "ratio",
            "unit_size_min",
            "unit_size_max",
            "holes_number_x",
            "holes_number_y",
            "shift_x",
            "shift_y",
            "mask_fill_value",
            "random_offset",
        )
