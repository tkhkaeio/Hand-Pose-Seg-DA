import torch
import torch.nn as nn
import numpy as np
import cv2

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_log(log, output_file, isPrint=True):
    with open(output_file, "a") as f:
        f.write(log + "\n")
        if isPrint:
            print(log)

def xyz2uvd(pts, paras, flip=1, scale_check=True):
    # paras: [fx, fy, fu, fv]
    pts_uvd = pts.copy()
    pts_uvd = pts_uvd.reshape(-1, 3)
    if scale_check:  # dexycb
        # pts_uvd[:, 2:] = pts_uvd[:, 2:] / 1000.
        pts_uvd = pts_uvd / 1000.
    pts_uvd[:, 1] *= flip  # flip y axis only
    # u, v <- x * fx / z + fu, y * fy / z + fv
    pts_uvd[:, :2] = pts_uvd[:, :2] * paras[:2] / pts_uvd[:, 2:] + paras[2:]

    if scale_check:
        pts_uvd[:, 2:] = pts_uvd[:, 2:] * 1000.
        # pts_uvd = pts_uvd * 1000.
    return pts_uvd.reshape(pts.shape).astype(np.float32)


def uvd2xyz(pts, paras, flip=1, scale_check=True):
    # paras: (fx, fy, fu, fv)
    pts_xyz = pts.copy()
    pts_xyz = pts_xyz.reshape(-1, 3)
    if scale_check:  # dexycb
        pts_xyz[:, 2:] = pts_xyz[:, 2:] / 1000.
        # pts_xyz = pts_xyz / 1000.
    pts_xyz[:, :2] = (pts_xyz[:, :2] - paras[2:]) * pts_xyz[:, 2:] / paras[:2]
    pts_xyz[:, 1] *= flip

    if scale_check:
        # pts_xyz[:, 2:] = pts_xyz[:, 2:] * 1000.
        pts_xyz = pts_xyz * 1000.
    return pts_xyz.reshape(pts.shape).astype(np.float32)


def draw_texts(img, texts, font_scale=0.7, thickness=2):
    h, w, c = img.shape
    offset_x = 20  # coord of left bottom
    initial_y = 50
    dy = int(img.shape[1] / 10)
    # color = (0, 0, 0)  # black
    # color = (0, 0, 200) #red
    color = (200, 0, 0) #blue

    texts = [texts] if type(texts) == str else texts

    for i, text in enumerate(texts):
        offset_y = initial_y + i*dy
        cv2.putText(img, text, (offset_x, offset_y), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, color, thickness, cv2.LINE_AA)

def draw_result_on_img(img, texts, w_ratio=0.255, h_ratio=0.17, alpha=0.4, font_scale=0.7, thickness=2):
    overlay = img.copy()
    pt1 = (0, 0)
    pt2 = (int(img.shape[1] * w_ratio), int(img.shape[0] * h_ratio))

    mat_color = (200, 200, 200)
    # mat_color = (256, 256, 256)
    fill = -1  # -1: fill
    cv2.rectangle(overlay, pt1, pt2, mat_color, fill)

    mat_img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    draw_texts(mat_img, texts, font_scale, thickness)

    return mat_img

def softargmax1d(input, beta=100):
    *_, n = input.shape
    input = nn.functional.softmax(beta * input, dim=-1)
    indices = torch.linspace(0, 1, n).to(input.device)
    result = torch.sum((n - 1) * input * indices, dim=-1)
    return result

def find_keypoints_max(heatmaps, soft_argmax=False):
    """
    heatmaps: C x H x W
    return: C x 3
    """
    # flatten the last axis
    heatmaps_flat = heatmaps.view(heatmaps.size(0), -1)

    # max loc
    if soft_argmax:
        max_ind = softargmax1d(heatmaps_flat)
    else:
        # max_val, max_ind = heatmaps_flat.max(1)
        _, max_ind = heatmaps_flat.max(1)
    max_ind = max_ind.float()

    max_v = torch.floor(torch.div(max_ind, heatmaps.size(1)))
    max_u = torch.fmod(max_ind, heatmaps.size(2))
    # return torch.cat((max_u.view(-1,1), max_v.view(-1,1), max_val.view(-1,1)), 1)
    return torch.cat((max_u.view(-1,1), max_v.view(-1,1)), 1)

def compute_uv_from_heatmaps(hm, resize_dim, soft_argmax=False):
    """
    :param hm: B x K x H x W (Variable)
    :param resize_dim:
    :return: uv in resize_dim (Variable)
    """
    upsample = nn.Upsample(size=resize_dim, mode='bilinear')  # (B x K) x H x W
    resized_hm = upsample(hm).view(-1, resize_dim[0], resize_dim[1])

    uv_confidence = find_keypoints_max(resized_hm, soft_argmax=soft_argmax)  # (B x K) x 3

    # jt_uvd = uv_confidence.view(-1, hm.size(1), 3)
    jt_uvd = uv_confidence.view(-1, hm.size(1), 2)
    jt_uvd[:, :, :2] = jt_uvd[:, :, :2] / (resize_dim[0] / 2) - 1
    return jt_uvd