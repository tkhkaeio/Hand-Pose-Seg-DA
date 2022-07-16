import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
import cv2
from enum import Enum

from scipy.linalg import orthogonal_procrustes
from helpers.util import draw_result_on_img
from helpers.util import uvd2xyz


class PoseEvalUtil(object):
    """ Util class for evaluation networks."""
    def __init__(self, img_size, flip, num_kp, dataset="ho3d"):
        # init empty data storage
        self.data = list()
        self.img_size = img_size
        self.flip = flip
        self.num_kp = num_kp
        self.dataset = dataset
        self.jt_uvd_pred = []
        self.diff = []
        for _ in range(num_kp):
            self.data.append(list())

    def feed_batch(self, jt_uvd_pred, jt_xyz_gt, center_xyz, M, cube, jt_vis=0,  jt_vis_mask=None, mode="abs", skip_check=False, is_center_crop=True, paras=None, score_return=False, jt_uvd_pred2=None, jt_uvd_gt=None, eval_2D=False):
        jt_uvd_pred = jt_uvd_pred.detach().cpu().numpy()
        self.dummpy_tensor = np.zeros_like(jt_uvd_pred)
        jt_xyz_gt = self.dummpy_tensor if jt_xyz_gt is None else jt_xyz_gt.cpu().numpy()
        if eval_2D: assert jt_uvd_gt is not None or jt_uvd_pred2 is not None
        jt_uvd_pred2 = self.dummpy_tensor if jt_uvd_pred2 is None else jt_uvd_pred2.detach().cpu().numpy()
        jt_uvd_gt = self.dummpy_tensor if jt_uvd_gt is None else jt_uvd_gt.clone().numpy()
        if jt_vis == 0: jt_vis = np.zeros(jt_uvd_pred.shape[0])
        jt_vis_mask = jt_vis_mask.cpu().numpy() if jt_vis_mask is not None else np.zeros(jt_uvd_pred.shape[0])

        center_xyz, M, cube, paras = center_xyz.numpy(), M.numpy(), cube.numpy(), paras.numpy()
        mpe_list, auc_list = list(), list()
        for i in range(jt_uvd_pred.shape[0]):
            mpe, auc = self.feed(jt_uvd_pred[i], jt_xyz_gt[i], center_xyz[i], M[i], cube[i], mode, jt_vis[i], jt_vis_mask[i], skip_check, is_center_crop, paras[i], score_return, jt_uvd_pred2[i], jt_uvd_gt[i], eval_2D)
            mpe_list.append(mpe)
            auc_list.append(auc)
        if score_return:
            return mpe_list, auc_list
        else:
            return mpe_list, None

    def feed(self, jt_uvd_pred, jt_xyz_gt, center_xyz, M, cube, mode="abs", jt_vis=0, jt_vis_mask=None, skip_check=False, is_center_crop=True, paras=None, score_return=False, jt_uvd_pred2=None, jt_uvd_gt=None, eval_2D=False):
        """ Used to feed data to the class. Stores the euclidean distance between gt and pred, when it is visible. """
        if eval_2D:
            jt_pred = (jt_uvd_pred[:, :2] + 1) * self.img_size / 2.
            if jt_uvd_pred2.sum() != 0: # pred2 exists -> compare btw/ predictions
                jt_gt = (jt_uvd_pred2[:, :2] + 1) * self.img_size / 2.
            else:
                jt_gt = (jt_uvd_gt[:, :2] + 1) * self.img_size / 2.
            self.err_max = 20 #px
            # print(jt_pred[0], jt_gt[0])
        else:
            if is_center_crop:
                if not skip_check:
                    jt_uvd_pred = np.squeeze(jt_uvd_pred).astype(np.float32)
                    jt_xyz_gt = np.squeeze(jt_xyz_gt).astype(np.float32)
                    jt_vis = np.squeeze(jt_vis).astype('bool')
                    center_xyz = np.squeeze(center_xyz).astype(np.float32)
                    M = np.squeeze(M).astype(np.float32)
                    cube = np.squeeze(cube).astype(np.float32)

                    assert len(jt_uvd_pred.shape) == 2
                    assert len(jt_xyz_gt.shape) == 2
                try:
                    M_inv = np.linalg.inv(M)
                except:
                    ('Inverse matrix does not exist.')

                jt_uvd_pred[:, :2] = (jt_uvd_pred[:, :2] + 1) * self.img_size / 2.
                jt_uvd_pred[:, 2] = jt_uvd_pred[:, 2] * cube[2] / 2. + center_xyz[2]
                jt_uvd_trans = np.hstack([jt_uvd_pred[:, :2], np.ones((jt_uvd_pred.shape[0], 1))])
                jt_uvd_pred[:, :2] = np.dot(M_inv, jt_uvd_trans.T).T[:, :2]
                self.jt_uvd_pred.append(jt_uvd_pred)
                if self.dataset=="fpha":
                    jt_xyz_pred = uvd2xyz(jt_uvd_pred, paras, self.flip, scale_check=False)
                else:
                    jt_xyz_pred = uvd2xyz(jt_uvd_pred, paras, self.flip)

                if jt_uvd_pred2.sum() != 0: # pred2 exists -> compare btw/ predictions
                    jt_uvd_pred2[:, :2] = (jt_uvd_pred2[:, :2] + 1) * self.img_size / 2.
                    jt_uvd_pred2[:, 2] = jt_uvd_pred2[:, 2] * cube[2] / 2. + center_xyz[2]
                    jt_uvd_trans = np.hstack([jt_uvd_pred2[:, :2], np.ones((jt_uvd_pred2.shape[0], 1))])
                    jt_uvd_pred2[:, :2] = np.dot(M_inv, jt_uvd_trans.T).T[:, :2]
                    if self.dataset=="fpha":
                        jt_xyz_pred2 = uvd2xyz(jt_uvd_pred2, paras, self.flip, scale_check=False)
                    else:
                        jt_xyz_pred2 = uvd2xyz(jt_uvd_pred2, paras, self.flip)
                    jt_xyz_gt = jt_xyz_pred2
                else:
                    jt_xyz_gt = jt_xyz_gt * (cube / 2.) + center_xyz
            else:
                if not skip_check:
                    jt_uvd_pred = np.squeeze(jt_uvd_pred).astype(np.float32)
                    jt_xyz_gt = np.squeeze(jt_xyz_gt).astype(np.float32)
                    jt_vis = np.squeeze(jt_vis).astype('bool')

                    assert len(jt_uvd_pred.shape) == 2
                    assert len(jt_xyz_gt.shape) == 2
                
                self.jt_uvd_pred.append(jt_uvd_pred)
                jt_xyz_pred = 0
            # jt_xyz_gt[:, :2], jt_xyz_pred[:, :2] = jt_xyz_gt[:,:2]*1000, jt_xyz_pred[:,:2]*1000
            jt_gt, jt_pred = jt_xyz_gt, jt_xyz_pred
            self.err_max = 50 #mm

        if mode=="abs":
            pass
        elif mode=="root":
            jt_pred = (jt_pred - jt_pred[0])
            jt_gt = (jt_gt - jt_gt[0])
        elif mode=="proc":
            jt_pred = align_w_scale(jt_gt, jt_pred)
        else:
            raise NotImplementedError()
        # calc euclidean distance on camera coordinate
        diff = jt_gt - jt_pred
        # print(diff)
        euclidean_dist = np.sqrt(np.sum(np.square(diff), axis=1))
        self.diff.append(diff.mean(axis=0))

        if score_return:
            auc_all = list()
            thresholds = np.linspace(0, self.err_max, 100)
            thresholds = np.array(thresholds)
            norm_factor = np.trapz(np.ones_like(thresholds), thresholds)

        num_kp = jt_xyz_gt.shape[0]
        for i in range(num_kp):
            # set mask region as visible
            if len(jt_vis_mask.shape)==3:
                jt_coord = jt_pred[i, :2].astype(np.int8)
                jt_vis = jt_vis_mask[0, jt_coord[1], jt_coord[0]]
                if int(jt_vis)==1:
                    self.data[i].append(euclidean_dist[i])
            else:
                if jt_vis == 0:
                    self.data[i].append(euclidean_dist[i])
                else:
                    if jt_vis[i]:
                        self.data[i].append(euclidean_dist[i])
            
            if score_return:
                pck_curve = list()
                for t in thresholds:
                    pck = np.mean((euclidean_dist[i] <= t).astype('float'))
                    pck_curve.append(pck)

                pck_curve = np.array(pck_curve)
                auc = np.trapz(pck_curve, thresholds)
                auc /= norm_factor
                auc_all.append(auc)
        
        if score_return:
            return euclidean_dist.mean(), np.mean(auc_all)
        else:
            return euclidean_dist.mean(), 0

    def _get_pck(self, kp_id, threshold):
        """ Returns pck for one keypoint for the given threshold. """
        if len(self.data[kp_id]) == 0:
            return None

        data = np.array(self.data[kp_id])
        pck = np.mean((data <= threshold).astype('float'))
        return pck

    def _get_epe(self, kp_id):
        """ Returns end point error for one keypoint. """
        if len(self.data[kp_id]) == 0:
            return None, None

        data = np.array(self.data[kp_id])
        epe_mean = np.mean(data)
        epe_median = np.median(data)
        return epe_mean, epe_median

    def get_measures(self):
        """ Outputs the average mean and median error as well as the pck score. """
        thresholds = np.linspace(0, self.err_max, 100)
        thresholds = np.array(thresholds)
        norm_factor = np.trapz(np.ones_like(thresholds), thresholds)

        # init mean measures
        epe_mean_all = list()
        epe_median_all = list()
        auc_all = list()
        pck_curve_all = list()

        # Create one plot for each part
        for part_id in range(self.num_kp):
            # mean/median error
            mean, median = self._get_epe(part_id)

            if mean is None:
                # there was no valid measurement for this keypoint
                continue

            epe_mean_all.append(mean)
            epe_median_all.append(median)

            # pck/auc
            pck_curve = list()
            for t in thresholds:
                pck = self._get_pck(part_id, t)
                pck_curve.append(pck)

            pck_curve = np.array(pck_curve)
            pck_curve_all.append(pck_curve)
            auc = np.trapz(pck_curve, thresholds)
            auc /= norm_factor
            auc_all.append(auc)

        # average mean/median error over num frames and joints
        epe_mean = np.mean(np.array(epe_mean_all))
        epe_median = np.mean(np.array(epe_median_all))
        # area under pck curve
        auc = np.mean(np.array(auc_all))
        pck_curve = np.mean(np.array(pck_curve_all), 0)  # mean only over keypoints
        return epe_mean, epe_median, auc, pck_curve, thresholds

    def plot_pck(self, path, pck_curve_all, thresholds):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(thresholds, pck_curve_all * 100, '-*', label='model')
        ax.set_xlabel('threshold in mm')
        ax.set_ylabel('% of correct keypoints')
        plt.ylim([0, 100])
        plt.grid()
        plt.legend(loc='lower right')
        # plt.tight_layout(rect=(01, -05, 1.03, 1.03))
        plt.savefig(path)
        plt.close()

def align_w_scale(mtx1, mtx2, return_trafo=False):
    """ Align the predicted entity in some optimality sense with the ground truth. """
    # center
    t1 = mtx1.mean(0)
    t2 = mtx2.mean(0)
    mtx1_t = mtx1 - t1
    mtx2_t = mtx2 - t2

    # scale
    s1 = np.linalg.norm(mtx1_t) + 1e-8
    mtx1_t /= s1
    s2 = np.linalg.norm(mtx2_t) + 1e-8
    mtx2_t /= s2

    # orth alignment
    R, s = orthogonal_procrustes(mtx1_t, mtx2_t)

    # apply trafos to the second matrix
    mtx2_t = np.dot(mtx2_t, R.T) * s
    mtx2_t = mtx2_t * s1 + t1
    if return_trafo:
        return R, s, s1, t1 - t2
    else:
        return mtx2_t

def get_sketch_setting():
    return [(0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20)]

class Color(Enum):
    RED = (0, 0, 255.0)
    GREEN = (75.0, 255.0, 66.0)
    BLUE = (255.0, 0, 0)
    YELLOW = (0, 186.0, 248.0)
    PURPLE = (255.0, 255.0, 0)
    CYAN = (255.0, 0, 255.0)
    BROWN = (204.0, 153.0, 17.0)


def get_sketch_color():
    return [Color.RED, Color.RED, Color.RED, Color.RED, Color.GREEN, Color.GREEN, Color.GREEN, Color.GREEN,
            Color.BLUE, Color.BLUE, Color.BLUE, Color.BLUE, Color.YELLOW, Color.YELLOW, Color.YELLOW, Color.YELLOW,
            Color.PURPLE, Color.PURPLE, Color.PURPLE, Color.PURPLE]

def get_joint_color():
    return [Color.CYAN, Color.RED, Color.RED, Color.RED, Color.RED, Color.GREEN, Color.GREEN, Color.GREEN, Color.GREEN,
            Color.BLUE, Color.BLUE, Color.BLUE, Color.BLUE, Color.YELLOW, Color.YELLOW, Color.YELLOW, Color.YELLOW,
            Color.PURPLE, Color.PURPLE, Color.PURPLE, Color.PURPLE]

def draw_2d_skeleton(image, pose_uv):
    """
    :param image: H x W x 3
    :param pose_uv: 21 x 2
    wrist,
    thumb_mcp, thumb_pip, thumb_dip, thumb_tip
    index_mcp, index_pip, index_dip, index_tip,
    middle_mcp, middle_pip, middle_dip, middle_tip,
    ring_mcp, ring_pip, ring_dip, ring_tip,
    little_mcp, little_pip, little_dip, little_tip
    :return:
    """
    assert pose_uv.shape[0] == 21
    skeleton_overlay = np.copy(image)
    color_hand_joints = get_joint_color()
    marker_sz = 3
    line_wd = 1
    root_ind = 0

    for joint_ind in range(pose_uv.shape[0]):
        joint = pose_uv[joint_ind, 0].astype('int32'), pose_uv[joint_ind, 1].astype('int32')
        # print("color", np.asarray(color_hand_joints[joint_ind].value) / 255.)
        cv2.circle(
            skeleton_overlay, joint,
            radius=marker_sz, color=color_hand_joints[joint_ind].value, thickness=-1,
            lineType=cv2.CV_AA if cv2.__version__.startswith('2') else cv2.LINE_AA)

        if joint_ind == 0:
            continue
        elif joint_ind % 4 == 1:
            root_joint = pose_uv[root_ind, 0].astype('int32'), pose_uv[root_ind, 1].astype('int32')
            cv2.line(
                skeleton_overlay, root_joint, joint,
                color=color_hand_joints[joint_ind].value, thickness=int(line_wd),
                lineType=cv2.CV_AA if cv2.__version__.startswith('2') else cv2.LINE_AA)
        else:
            joint_2 = pose_uv[joint_ind - 1, 0].astype('int32'), pose_uv[joint_ind - 1, 1].astype('int32')
            cv2.line(
                skeleton_overlay, joint_2, joint,
                color=color_hand_joints[joint_ind].value, thickness=int(line_wd),
                lineType=cv2.CV_AA if cv2.__version__.startswith('2') else cv2.LINE_AA)

    return skeleton_overlay

def draw_pose(dataset, img, pose):
    # assert dataset in ['icvl', 'nyu', 'nyu_full', 'msra', 'hands17', 'itop', 'ho3d', 'dexycb', 'fpha', 'hanco', 'rhd']
    colors = get_sketch_color()
    colors_joint = get_joint_color()
    idx = 0
    for pt in pose:
        cv2.circle(img, (int(pt[0]), int(pt[1])), 3, colors_joint[idx].value, -1)
        idx = idx + 1
    idx = 0
    for x, y in get_sketch_setting():
        cv2.line(img, (int(pose[x, 0]), int(pose[x, 1])),
                 (int(pose[y, 0]), int(pose[y, 1])), colors[idx].value, 1)
        idx = idx + 1
    return img

def debug_2d_pose(img, joint_img, index, save_dir, save_name, dataset="ho3d", img_name=None, M=None, score=None, score2=None, is_color=False, is_center_origin=True, is_mask=False):
    batch_size, _, _, input_size = img.size()
    # index = index - batch_size + 1
    os.makedirs(osp.join(save_dir, f"fig/{save_name}"), exist_ok=True)
    if is_center_origin:
        jt_uvd = (joint_img + 1) / 2 * input_size
        # print(jt_uvd[0]) [[ 80.3175,  81.9956,  46.3996],
    else:
        jt_uvd = joint_img
        jt_uvd[:, 0] = jt_uvd[:, 0]
        jt_uvd[:, 1] = jt_uvd[:, 1]
        jt_uvd[:, :, 2] = (joint_img[:, :, 2] + 1) / 2 * input_size
    for j in range(batch_size):
        if not is_mask:
            img_draw = (img.detach().cpu().numpy() + 1) / 2 * 255.0
        else:
            img_draw = img.detach().cpu().numpy() * 255.0
        if is_color:
            img_show = draw_pose(dataset, np.ascontiguousarray(np.transpose(img_draw[j], (1, 2, 0)), dtype=np.uint8), jt_uvd[j])
            # img_show = draw_pose(dataset, np.transpose(img_draw[j], (1, 2, 0)).copy(), jt_uvd[j])
        else:
            img_show = draw_pose(dataset, cv2.cvtColor(img_draw[j, 0], cv2.COLOR_GRAY2BGR), jt_uvd[j])
        img_show = cv2.resize(img_show, (500, 500))
        if score is not None:
            if len(img_show.shape)==2:
                img_show = np.expand_dims(img_show, axis=2)
            if score2 is not None:
                img_show = draw_result_on_img(img_show, [f"MPE: {score[j]:.2f}", f"MPE(2d): {score2[j]:.2f}"], w_ratio=0.8, h_ratio=0.25, alpha=0.6, font_scale=1.5, thickness=6)
            else:
                img_show = draw_result_on_img(img_show, f"MPE: {score[j]:.2f}", w_ratio=0.7, h_ratio=0.15, alpha=0.6, font_scale=1.5, thickness=6)
        if img_name is None:
            filename = f"{index * batch_size + j:05d}"
        else:
            filename = img_name[j].rsplit(".", 1)[0]
        cv2.imwrite(osp.join(save_dir, f"fig/{save_name}/{save_name}_{filename}.png"), img_show)
    # return img_show