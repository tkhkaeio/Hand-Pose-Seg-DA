from tqdm import tqdm

import torch
import torch.nn as nn
import hydra
from omegaconf import DictConfig

from run_base import BaseTrainer
from helpers.util import AverageMeter, save_log
from helpers.pose_tool import PoseEvalUtil
from helpers.seg_tool import seg_eval
from dataloader.transformation import TestTimeAugmentation

class Trainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)
        # update params passing to base trainer
        self.load_model = self.cfg.model.load_joint_model
        self.load_pretrain_model = None
        self.net = self.cfg.model.joint_net
        self.batch_size = self.cfg.base.batch_size
        self.lr = self.cfg.solver.joint_lr
        self.scheduler_type = self.cfg.solver.joint_scheduler
        assert self.cfg.base.adapt_mode in ["none", 
                                            "gac",
                                            "gac_freeze_mask",
                                            ]
        # initialization of adaptation models
        if self.cfg.base.mode == "train" and self.cfg.base.adapt_mode != "none":
            self.load_pretrain_model = self.cfg.pretrain.load_joint_model_rgb
            

        self.model_type = self.net

        self.logging()
        self.model, self.FM = self.setup_model()
        self.model = self.model.to(self.device)
                
        self.criterion_pose, self.criterion_seg = self.setup_losses()
        self.criterion_mse = nn.MSELoss(reduction='none').to(self.device)
        self.mse_scale = 5
        self.criterion_seg2 = nn.BCELoss(reduction='none').to(self.device)
        # set optimizers
        self.optimizer, self.scheduler = self.setup_optimizer(self.model, self.lr, self.cfg.solver.beta1)
        # set data loader
        if self.cfg.base.adapt_mode != "none": # cross-domain
            save_log(f"adaptation mode: {self.cfg.base.adapt_mode}", self.log_path)
            self.train_s_loader, self.train_t_loader, self.val_loader, self.test_loader, max_train_s_size, max_train_t_size = self.setup_datasets(train_s=self.cfg.base.source_dataset, train_t=self.cfg.base.target_dataset, val_test=self.cfg.base.target_dataset)
        else:                                  # single domain
            if self.cfg.base.mode == "train":
                self.train_s_loader, _, self.val_loader, self.test_loader, max_train_s_size, max_train_t_size = self.setup_datasets(train_s=self.cfg.base.dataset, train_t=None, val_test=self.cfg.base.dataset)
            else:
                _, _, self.val_loader, self.test_loader, max_train_s_size, max_train_t_size = self.setup_datasets(train_s=None, train_t=None, val_test=self.cfg.base.dataset)

        self.max_train_size = max(max_train_s_size, max_train_t_size)
        self.max_train_t_size = max_train_t_size

    def train_adapt(self, epoch):
        self.model.train()
        
        self.pose_eval_tool = PoseEvalUtil(self.cfg.base.img_size, self.train_s_loader.dataset.flip, self.cfg.base.jt_num, self.eval_dataset)
        weak_aug = TestTimeAugmentation("weak", self.cfg.base.modality)
        loss_meter = AverageMeter()
        unsup_data_meter = AverageMeter()
        task_loss_meter = AverageMeter()
        adapt_loss_meter = AverageMeter()
        n_iters = 0
        train_iter_s = iter(self.train_s_loader)
        train_iter_t = iter(self.train_t_loader)

        for _ in tqdm(range(self.max_train_size//self.batch_size)):
            try:
                data_s = next(train_iter_s)
            except StopIteration:
                train_iter_s = iter(self.train_s_loader)
                data_s = next(train_iter_s)
            try:
                data_t = next(train_iter_t)
            except StopIteration:
                train_iter_t = iter(self.train_t_loader)
                data_t = next(train_iter_t)

            img_s, mask_s, jt_uvd_gt_s, jt_uvbgr_gt_s, jt_valid_s = data_s["img"].to(self.device), data_s["mask"].to(self.device), data_s["jt_uvd_gt"].to(self.device), data_s["jt_uvbgr_gt"].to(self.device), data_s["jt_valid"].to(self.device)
            img_t = data_t["img"].to(self.device)
            n_iters += img_s.size(0)

            self.optimizer.zero_grad()
            
            adapt_loss = torch.zeros(1).to(self.device)
            if "gac" in self.cfg.base.adapt_mode:
                keypoints = torch.cat([jt_uvd_gt_s.cpu().clone(), jt_uvbgr_gt_s.cpu().clone()], dim=-1)
                img_s_aug, keypoints_aug, mask_s_aug = weak_aug(img_s.cpu(), keypoints, mask_s.cpu().clone())
                jt_uvd_gt_s_aug = keypoints_aug[:, :, :3].contiguous().to(self.device)
                jt_uvbgr_gt_s_aug = keypoints_aug[:, :, 3:].contiguous().to(self.device)
                    
                img_s_aug = img_s_aug.to(self.device)
                mask_s_aug = mask_s_aug.to(self.device)
                
                if self.cfg.base.debug: print("student: source task loss")
                jt_uvd_pred_s_aug, offset_pred_s_aug, seg_pred_s_aug = self.model(img_s_aug)
                task_loss = self._compute_task_loss(jt_uvd_pred_s_aug, offset_pred_s_aug, seg_pred_s_aug, jt_uvd_gt_s_aug, jt_uvbgr_gt_s_aug, img_s_aug, mask_s_aug, jt_valid_s)

                # prediction on no aug data_t & use it for pseudo-GT
                with torch.no_grad():
                    jt_uvd_pred_t, offset_pred_t, seg_pred_t = self.model(img_t)
                # pl_mask_t = (seg_pred_t.detach() > 0.5).float()
                
                unsup_data_meter.update(jt_uvd_pred_t.size(0), 1)
                
                img_t_aug, pl_jt_uvd_t_aug, pl_seg_pred_t_aug = weak_aug(img_t.cpu(), jt_uvd_pred_t.cpu().detach().clone(), seg_pred_t.cpu().detach().clone())
                
                img_t_aug = img_t_aug.to(self.device)
                pl_jt_uvd_t_aug = pl_jt_uvd_t_aug.to(self.device)
                pl_seg_pred_t_aug = pl_seg_pred_t_aug.to(self.device)
                # pl_mask_t_aug = (pl_seg_pred_t_aug > 0.5).float()

                # prediction on aug data_t & be consistent with pseudo-GT
                if self.cfg.base.debug: print("target unsupervised loss")
                jt_uvd_pred_t_aug, offset_pred_t_aug, seg_pred_t_aug = self.model(img_t_aug, freeze_mask_head=("freeze_mask" in self.cfg.base.adapt_mode))
                
                adapt_loss = self.cfg.loss.unsup_coef * self._compute_task_loss(jt_uvd_pred_t_aug, offset_pred_t_aug, seg_pred_t_aug, pl_jt_uvd_t_aug, None, img_t_aug, pl_seg_pred_t_aug, is_PL=True)
            else:
                raise NotImplementedError()
            

            loss = task_loss + adapt_loss
            loss_meter.update(loss.item(), self.batch_size)
            task_loss_meter.update(task_loss.item(), self.batch_size)
            adapt_loss_meter.update(adapt_loss.item(), self.batch_size)
            loss.backward()

            self.optimizer.step()
            
            if self.cfg.base.debug: break
            if n_iters > 5_000: break

        cur_lr = self.optimizer.param_groups[0]['lr']
        train_report = f"train    ({self.cfg.base.modality}) [epoch {epoch:02d}][Loss {loss_meter.avg:.3f}][Task Loss {task_loss_meter.avg:.3f}][DA Loss {adapt_loss_meter.avg:.3f}][lr {cur_lr:.2e}]"
        save_log(train_report, self.log_path)
        if "gac" in self.cfg.base.adapt_mode:
            save_log(f"#unlabeled instance per epoch: {unsup_data_meter.sum:.1f} / {n_iters}", self.log_path)

        if self.scheduler_type == 'auto':
            self.scheduler.step(self.val_score)
            save_log(f"auto scheduler's score: {self.val_score:.1f}", self.log_path)
        elif self.scheduler_type != "none":
            self.scheduler.step()
        
    @torch.no_grad()
    def evaluate(self, epoch, loader, model, mode="val", full_eval=False, save_all=False, save_name=None):
        model.eval()
        device = model.dummy_param.device
        if self.eval_scores:
            if self.cfg.base.pose_2d:
                pose_eval_tool = PoseEvalUtil(self.cfg.base.img_size, loader.dataset.flip, self.cfg.base.jt_num, self.eval_dataset)
            iou_meter = AverageMeter()
        loss_meter = AverageMeter()
        n_iters = 0
        for i, data in tqdm(enumerate(loader)):
            img, mask, jt_uvd, jt_uvbgr, jt_valid = data["img"].to(device), data["mask"].to(device), data["jt_uvd_gt"].to(device), data["jt_uvbgr_gt"].to(device), data["jt_valid"].to(device)
            n_iters += img.size(0)
            
            loss, jt_uvd_pred, offset_pred, seg_pred = self._forward(img, jt_uvd, jt_uvbgr, mask, jt_valid, model=model)
            loss_meter.update(loss.item(), self.cfg.base.val_test_batch_size)
            
            if self.eval_scores:
                _mpe_list, _auc_list = pose_eval_tool.feed_batch(jt_uvd_pred, data["jt_xyz_gt"], data["center_xyz"], data["M"], data["cube"], paras=data["paras"], score_return=True, jt_uvd_gt=data["jt_uvd_gt"], eval_2D=self.cfg.base.pose_2d)
                _iou, _, n_seg_eval, _ = seg_eval(seg_pred, mask, with_kappa=False, is_single_hand=(self.eval_dataset=="fpha"))
                if self.cfg.base.debug: print("iou", _iou)
                if n_seg_eval > 0:
                    iou_meter.update(_iou*100, n_seg_eval)
            
            # when to stop evaluation
            if self.cfg.base.debug: break
            else: 
                if (not full_eval) and n_iters > 10_000: break

        val_score = 0
        ret_report = "none"
        if self.eval_scores:
            iou = iou_meter.avg
            if self.cfg.base.pose_2d:
                mpe, _, auc, _, _ = pose_eval_tool.get_measures()
                auc *= 100
                save_log(f"{mode:<8} ({self.eval_dataset}/{self.cfg.base.modality}) [epoch {epoch:02d}][Loss {loss_meter.avg:.3f}][MPE {mpe:.3f}][AUC {auc:.3f}][IoU {iou:.3f}][lr {self.optimizer.param_groups[0]['lr']:.2e}]", self.log_path)
                ret_report = f"best [epoch {epoch:02d}][MPE {mpe:.3f}][AUC {auc:.3f}][IoU {iou:.3f}][Avg {(auc + iou)/2:.3f}]"
                if "freeze_mask" in self.cfg.base.adapt_mode:
                    val_score = auc - mpe
                else:
                    if self.cfg.loss.seg_weight == 0: 
                        val_score =  auc - mpe
                    elif self.cfg.loss.dense_2d_weight == 0:
                        val_score = iou
                    else:
                        val_score = auc - mpe + iou

            if mode=="val":
                self.val_score = val_score
            elif mode=="val2":
                self.val_score2 = val_score
        return ret_report, val_score

    def _compute_task_loss(self, jt_uvd_pred, offset_pred, seg_pred, jt_uvd, jt_uvbgr, img, mask, jt_valid=None, is_PL=False, loss_select=None, uct_score=None):
        if loss_select is not None:
            assert is_PL
            if self.cfg.base.pose_2d:
                loss_seg = self.cfg.loss.seg_weight * self.criterion_seg2(seg_pred, mask).mean((1,2,3))

                offset_gt = self.FM.joint2offset(jt_uvd, jt_uvd, img)
                loss_offset = self.cfg.loss.dense_2d_weight * self.criterion_pose(offset_pred, offset_gt, reduce=False).mean((1,2,3))
                loss_pose = loss_offset
                uct_weight = 2 * (1 - torch.sigmoid(self.cfg.da_para.sigmoid_weight * loss_select.to(seg_pred.device)))
                if self.cfg.base.debug: print("uct range", uct_weight.min(), uct_weight.mean(), uct_weight.max())
                loss = (loss_pose + loss_seg) * uct_weight
                loss = loss.mean()
            else:
                raise NotImplementedError()
        elif uct_score is not None:
            assert is_PL
            if self.cfg.base.pose_2d:
                loss_seg = self.cfg.loss.seg_weight * self.criterion_seg2(seg_pred, mask).mean((1,2,3))

                offset_gt = self.FM.joint2offset(jt_uvd, jt_uvd, img)
                loss_offset = self.cfg.loss.dense_2d_weight * self.criterion_pose(offset_pred, offset_gt, reduce=False).mean((1,2,3))
                loss_pose = loss_offset
                uct_weight = 1 - uct_score
                loss = (loss_pose + loss_seg) * uct_weight
                loss = loss.mean()
            else:
                raise NotImplementedError()
        else:
            loss_seg = self.cfg.loss.seg_weight * self.criterion_seg(seg_pred, mask)
            if self.cfg.base.pose_2d:
                if is_PL: # backprop coordinate level only 
                    offset_gt = self.FM.joint2offset(jt_uvd, jt_uvd, img)
                    loss_offset = self.cfg.loss.dense_2d_weight * self.criterion_pose(offset_pred, offset_gt)
                    loss_pose = loss_offset
                else: # heatmap loss
                    offset_gt = self.FM.joint2offset(jt_uvd, jt_uvbgr, img)
                    loss_offset = self.cfg.loss.dense_2d_weight * self.criterion_pose(offset_pred, offset_gt, jt_valid)
                    loss_pose = loss_offset
            if self.cfg.base.debug: print("pose", loss_pose.item(), "seg", loss_seg.item())

            loss = (loss_pose + loss_seg)
        return loss
    
    def _forward(self, img, jt_uvd, jt_uvbgr, mask, jt_valid, pretrain=False, model=None):
        if model is None:
            model = self.model
        if pretrain:
            if "hourglass" in self.net:
                offset_gt = self.FM.joint2offset(jt_uvd, jt_uvbgr, img)
                jt_uvd_pred, offset_pred, seg_pred = model(img)
                loss = self.cfg.loss.dense_2d_weight * self.criterion_pose(offset_pred, offset_gt, jt_valid)
        else:
            if "hourglass" in self.net:
                jt_uvd_pred, offset_pred, seg_pred = model(img)
                loss = self._compute_task_loss(jt_uvd_pred, offset_pred, seg_pred, jt_uvd, jt_uvbgr, img, mask, jt_valid)
                if self.cfg.base.debug: print("pred coord scale", jt_uvd_pred[0][0].detach().cpu().numpy(), jt_uvd[0][0].cpu().numpy())
        
        return loss, jt_uvd_pred, offset_pred, seg_pred
    
    def train(self, epoch):
        self.model.train()
        pose_eval_tool = PoseEvalUtil(self.cfg.base.img_size, self.train_s_loader.dataset.flip, self.cfg.base.jt_num, self.eval_dataset)
        iou_meter = AverageMeter()
        loss_meter = AverageMeter()
        n_iters = 0
        for i, data in tqdm(enumerate(self.train_s_loader)):
            img, mask, jt_uvd, jt_uvbgr, jt_valid = data["img"].to(self.device), data["mask"].to(self.device), data["jt_uvd_gt"].to(self.device), data["jt_uvbgr_gt"].to(self.device), data["jt_valid"].to(self.device)
            n_iters += img.size(0)
            self.optimizer.zero_grad()
            
            loss, jt_uvd_pred, offset_pred, seg_pred = self._forward(img, jt_uvd, jt_uvbgr, mask, jt_valid)

            loss_meter.update(loss.item(), self.cfg.base.batch_size)
            loss.backward()
            self.optimizer.step()

            pose_eval_tool.feed_batch(jt_uvd_pred, data["jt_xyz_gt"], data["center_xyz"], data["M"], data["cube"], paras=data["paras"], jt_uvd_gt=data["jt_uvd_gt"], eval_2D=self.cfg.base.pose_2d)
            _iou, _, _, _ = seg_eval(seg_pred, mask)
            iou_meter.update(_iou*100, self.cfg.base.batch_size)

            if self.cfg.base.debug: break
            if n_iters > 50_000: break

        mpe, mid, auc, pck, thresh = pose_eval_tool.get_measures()
        cur_lr = self.optimizer.param_groups[0]['lr']
        save_log(f"train    ({self.eval_dataset}/{self.cfg.base.modality}) [epoch {epoch:02d}][Loss {loss_meter.avg:.3f}][MPE {mpe:.3f}][AUC {auc*100:.3f}][IoU {iou_meter.avg:.3f}][lr {cur_lr:.2e}]", self.log_path)
        
        if self.scheduler_type == 'auto':
            self.scheduler.step(self.val_score)
        elif self.scheduler_type != "none":
            self.scheduler.step()
    
@hydra.main(config_name="config_base", config_path="config")
def main(config: DictConfig):
    trainer = Trainer(config)
    # this code support for 2d pose prediction only
    # add evaluation metrics and their significant digits
    trainer.run([{"AUC": 1, "MPE": 2, "IoU": 1, "Avg": 1},])

if __name__=='__main__':
    main()
