import numpy as np
import warnings
from sklearn.metrics import cohen_kappa_score

warnings.resetwarnings()
warnings.simplefilter('ignore', UserWarning)

def _get_first_index(arr):
    for i in range(arr.shape[0]):
        if arr[i]:
            return i
    return arr.shape[0]

def seg_eval(pred, mask, thresh=0.5, with_kappa=False, is_single_hand=False):
    thresh_value = 256 * thresh
    # compute IoU
    scores_list = {"miou": [], "kappa": [], "intersection": []}
    for i in range(pred.size(0)):
        if mask.sum()==0:
            scores_list["miou"].append(0)
            scores_list["intersection"].append(0)
            if with_kappa:
                scores_list["kappa"].append(0)
            continue
        # compute segmentation map from probability map
        pred_numpy = np.clip(np.transpose(pred.data[i].cpu().numpy(), (1, 2, 0)) * 255, 0, 255).astype(np.uint8)
        mask_numpy = np.clip(np.transpose(mask.data[i].cpu().numpy(), (1, 2, 0)) * 255, 0, 255).astype(np.uint8)
        if is_single_hand:
            mask_x = mask_numpy.sum(0) > 0
            x_left = max(0, _get_first_index(mask_x) - 10)
            x_right = min(mask_x.shape[0] - _get_first_index(mask_x[::-1]) + 10, mask_x.shape[0])
            mask_y = mask_numpy.sum(1) > 0
            y_top = max(0, _get_first_index(mask_y) - 10)
            y_bottom = min(mask_y.shape[0] - _get_first_index(mask_y[::-1]) + 10, mask_y.shape[0])

            pred_numpy = pred_numpy[x_left:x_right, y_top:y_bottom, :]
            mask_numpy = mask_numpy[x_left:x_right, y_top:y_bottom, :]

        pred_binary = (pred_numpy>thresh_value).reshape(-1).astype(np.uint8)
        mask_binary = (mask_numpy>thresh_value).reshape(-1).astype(np.uint8)
        union = np.logical_or(mask_binary, pred_binary)
        intersection = np.logical_and(mask_binary, pred_binary)
        iou = np.count_nonzero(intersection) / (np.count_nonzero(union) + 1e-7)
        scores_list["miou"].append(iou)
        scores_list["intersection"].append(intersection.reshape(pred_numpy.shape[:2]))
        if with_kappa:
            kappa = cohen_kappa_score(mask_binary, pred_binary)
            scores_list["kappa"].append(kappa)


    miou = np.sum(scores_list["miou"]) / (np.nonzero(scores_list["miou"])[0].size + 1e-7)
    kappa = np.sum(scores_list["kappa"]) / (np.nonzero(scores_list["kappa"])[0].size + 1e-7) if with_kappa else 0
    num = np.nonzero(scores_list["miou"])[0].size
    return miou, kappa, num, scores_list