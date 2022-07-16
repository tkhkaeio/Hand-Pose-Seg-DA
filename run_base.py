import os
import os.path as osp
import sys
import datetime
import re

from omegaconf import OmegaConf
import hydra
import albumentations as A

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

from models.hourglass import get_hourglass_net
from models.loss import SmoothL1Loss
from dataloader.dexycb_loader import DexYCB
from dataloader.ho3d_loader import HO3D
from helpers.util import save_log
from helpers.feature_tool import FeatureModule

class BaseTrainer(object):
    def __init__(self, config):
        self.cfg = config.exp
        self.cwd = hydra.utils.get_original_cwd()
        print("self.cfg.base.mode", self.cfg.base.mode)
        assert self.cfg.base.mode in ["train", "val", "test", "val_test"]
        assert osp.exists(self.cfg.base.data_root)
        if not osp.exists(self.cfg.base.output_dir):
            os.makedirs(self.cfg.base.output_dir)

        self.val_score = -float("inf")
        # params should be initialized by traniner
        self.model_type = None
        self.load_model = None
        self.load_pretrain_model = None
        self.net = None
        self.batch_size = None
        self.lr = None
        if self.cfg.base.adapt_mode != "none":
            self.eval_dataset = self.cfg.base.target_dataset
        else:
            self.eval_dataset = self.cfg.base.dataset
        # when labels for evaluation is available
        self.eval_scores = True

        torch.cuda.set_device(self.cfg.base.gpu_id)
        cudnn.benchmark = True
        self.device=f"cuda:{self.cfg.base.gpu_id}"


    def logging(self):
        # check output path
        if osp.exists(self.load_model) and self.load_model.endswith("pth"):
            log_path_eval = "/".join(self.load_model.split("/")[:-2])
        
        # output dirs for model, log and result figure saving
        now = datetime.datetime.now().strftime("%y%m%d_%H-%M-%S")
        if self.cfg.base.mode == "train":
            if self.cfg.base.debug:
                self.log_dir = f"{self.cwd}/debug"
            else:
                if self.cfg.base.adapt_mode != "none": # cross-domain
                    self.log_dir = osp.join(self.cfg.base.output_dir, f"{now}_{self.cfg.base.adapt_mode}_{self.cfg.base.modality}_{self.cfg.base.source_dataset[0]}2{self.cfg.base.target_dataset[0]}_{self.net}")
                else:                                  # single-domain
                    self.log_dir = osp.join(self.cfg.base.output_dir, f"{now}_{self.cfg.base.modality}_{self.cfg.base.dataset}_{self.net}")
            self.log_path = osp.join(self.log_dir, 'log.txt')
            self.model_save = osp.join(self.log_dir, 'checkpoint')
            if not osp.exists(self.model_save):
                os.makedirs(self.model_save)
        else:
            self.log_dir = log_path_eval
            self.log_path = osp.join(self.log_dir, f'val_test_{self.eval_dataset}.txt')
            assert osp.exists(self.load_model) and self.load_model.endswith("pth")
        self.result_dir = osp.join(self.log_dir, 'results')
        self.cfg_path = osp.join(self.log_dir, 'config.txt')
        if not osp.exists(self.result_dir):
            os.makedirs(self.result_dir)
        save_log(f"time: {now}", self.log_path)
        save_log(f"save log to {self.log_dir}", self.log_path)
        save_log(f"pytorch version: {torch.__version__}", self.log_path)
        save_log(f"python version: {sys.version}", self.log_path)
        if self.cfg.base.mode == "train":
            save_log(OmegaConf.to_yaml(self.cfg), self.cfg_path)
        else:
            print(OmegaConf.to_yaml(self.cfg))

    def setup_model(self, no_model_load=False):
        in_channels = 3 if self.cfg.base.modality == "rgb" else 1
        downsample = 1 if ("seg" in self.model_type or  "joint" in self.model_type) else self.cfg.model.downsample
        self.feat_size = self.cfg.base.img_size // downsample
        if "hourglass" in self.net:
            model = get_hourglass_net(self.model_type, self.cfg.base.jt_num, in_channels=in_channels, num_stages=2, num_modules=2, feat_size=self.feat_size // 2, use_dropout=True)
        else:
            raise NotImplementedError()

        # if not self.cfg.base.debug:
        if self.cfg.base.mode == "train" and self.load_pretrain_model is not None:
            if osp.exists(self.load_pretrain_model) and self.load_pretrain_model.endswith("pth"):
                pth = torch.load(self.load_pretrain_model)
                new_pth = {}
                save_log(f"loading pretrained model from {self.load_pretrain_model}", self.log_path)
                for k, v in model.state_dict().items():
                    if "dummy" in k:
                        new_pth[k] = v
                    elif "hm_head" in k:
                        new_pth[k] = v
                    else:
                        new_pth[k] = pth[k]
                # new_pth = {k:v for k, v in pth.items() if not "dummy" in k}
                model.load_state_dict(new_pth)

        if osp.exists(self.load_model) and self.load_model.endswith("pth") and not no_model_load:
            save_log(f"loading model from {self.load_model}", self.log_path)
            pth = torch.load(self.load_model, map_location="cpu")
            new_pth = {}
            for k, v in model.state_dict().items():
                if "dummy" in k:
                    new_pth[k] = v
                else:
                    new_pth[k] = pth[k]
            model.load_state_dict(new_pth)
        else:
            save_log("training from scatch", self.log_path)

        # use for pose estimation
        hm_2d = True if "hourglass" in self.net else self.cfg.base.pose_2d
        FM = FeatureModule(self.feat_size//2, self.cfg.base.modality, self.cfg.base.jt_num, self.cfg.model.kernel_size, self.cfg.model.heatmap_std, is_2D=hm_2d)
        return model, FM
    
    def _get_dataset(self, dataset_name, phase, aug_type="none"):
        if dataset_name=="dexycb":
            dataset = DexYCB(osp.join(self.cfg.base.data_root, "DexYCB"), phase, modality=self.cfg.base.modality, img_size=self.cfg.base.img_size, cube=self.cfg.base.cube, cwd=self.cwd, aug_type=aug_type)
        elif dataset_name=="ho3d":
            dataset = HO3D(osp.join(self.cfg.base.data_root, "HO3D/HO3D_v3"), phase, modality=self.cfg.base.modality, img_size=self.cfg.base.img_size, cube=self.cfg.base.cube, cwd=self.cwd, aug_type=aug_type)
        else:
            raise NotImplementedError()
        return dataset

    def setup_datasets(self, train_s="dexycb", train_t="dexycb", val_test="dexycb", t_sequence=False):
        aug_type_s = aug_type_t = "none"
        if self.cfg.base.aug_type != "none":
            aug_type_s = self.cfg.base.aug_type
            aug_type_t = self.cfg.base.aug_type
        if self.cfg.base.aug_type_s != "none":
            aug_type_s = self.cfg.base.aug_type_s
        if self.cfg.base.aug_type_t != "none":
            aug_type_t = self.cfg.base.aug_type_t
        save_log(f"aug type source: {aug_type_s}, target: {aug_type_t}", self.log_path)

        if train_s is not None:
            train_s_set = self._get_dataset(train_s, "train", aug_type_s)
            save_log(f"source domain    : {train_s: <8} data size: {train_s_set.__len__()}", self.log_path)
            if self.cfg.base.mode == "train":
                A.save(train_s_set.transform, osp.join(self.log_dir, 'transform_source.json'))
                # save_log(f"training source augmentation: {train_s_set.transform}", )
            max_train_s_size = len(train_s_set)
            train_s_loader = DataLoader(train_s_set, batch_size=self.batch_size, shuffle=True, num_workers=self.cfg.base.num_workers)
        else:
            train_s_loader = None
            max_train_s_size = 0
        if train_t is not None:
            train_t_set = self._get_dataset(train_t, "train", aug_type_t)
            save_log(f"target domain    : {train_t: <8} data size: {train_t_set.__len__()}", self.log_path)
            if self.cfg.base.mode == "train":
                A.save(train_t_set.transform, osp.join(self.log_dir, 'transform_target.json'))
                # save_log(f"training target augmentation: {train_t_set.transform}", self.log_path)
            max_train_t_size = len(train_t_set)
            if t_sequence:
                save_log("no shuffle in the target dataset", self.log_path)
                train_t_loader = DataLoader(train_t_set, batch_size=self.batch_size, shuffle=False, num_workers=self.cfg.base.num_workers)
            else:
                train_t_loader = DataLoader(train_t_set, batch_size=self.batch_size, shuffle=True, num_workers=self.cfg.base.num_workers)
        else:
            train_t_loader = None
            max_train_t_size = 0

        val_set = self._get_dataset(val_test, "val")
        test_set = self._get_dataset(val_test, "test")
        save_log(f"validation domain: {val_test: <8} data size: {val_set.__len__()}", self.log_path)
        save_log(f"testing domain   : {val_test: <8} data size: {test_set.__len__()}", self.log_path)
        val_loader = DataLoader(val_set, batch_size=self.cfg.base.val_test_batch_size, shuffle=self.cfg.base.eval_shuffle, num_workers=self.cfg.base.val_test_num_workers)
        test_loader = DataLoader(test_set, batch_size=self.cfg.base.val_test_batch_size, shuffle=self.cfg.base.eval_shuffle, num_workers=self.cfg.base.val_test_num_workers)
        
        return train_s_loader, train_t_loader, val_loader, test_loader, max_train_s_size, max_train_t_size
    
    def setup_losses(self):
        criterion_pose = SmoothL1Loss()
        criterion_seg = nn.BCELoss()

        return criterion_pose, criterion_seg
    
    def setup_optimizer(self, model, lr, beta1=0.5):
        optimizer = Adam(model.parameters(), lr=lr, betas=(beta1, 0.999), weight_decay=0.0005)
        if self.scheduler_type == 'auto':
            scheduler = ReduceLROnPlateau(optimizer, "max", factor=0.5, patience=2, min_lr=1e-8)
        elif self.scheduler_type == 'step':
            scheduler = StepLR(optimizer, step_size=self.cfg.solver.step_epoch, gamma=0.4)
        else:
            scheduler = None

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            save_log(f"learning rate: {param_group['lr']:.3e}", self.log_path)
        
        return optimizer, scheduler

    def train(self, epoch):
        pass

    @torch.no_grad()
    def evaluate(self, epoch, loader, model, mode="val", full_eval=False):
        pass

    def final_test(self, epoch, score_format=None):
        # load the best models
        if not self.cfg.base.debug and self.cfg.base.mode == "train":
            save_log("load the best model", self.log_path)
            self.model = self.model.cpu()
            pth = torch.load(osp.join(self.model_save, 'net_latest.pth'))
            self.model.load_state_dict(pth)
            self.model = self.model.to(self.device)
        # run validation and testing
        if self.cfg.base.mode == "train" or "val" in self.cfg.base.mode or "test" in self.cfg.base.mode:
            # validation
            if self.cfg.base.mode == "train" or "val" in self.cfg.base.mode:
                self.model.eval()
                val_report, _ = self.evaluate(epoch, loader=self.val_loader, model=self.model, full_eval=True, mode="val")
            # testing
            if self.cfg.base.mode == "train" or "test" in self.cfg.base.mode:
                self.model.eval()
                test_report, _ = self.evaluate(epoch, loader=self.test_loader, model=self.model, full_eval=True, mode="test")    
             # formatting results
            if score_format:
                # save_log("\nresult (val / test)", self.log_path)
                for i, _score_format in enumerate(score_format):
                    ret_values = ""
                    ret_keys = ""
                    assert len(val_report.split("\n")) == len(score_format)
                    for k, v in _score_format.items():
                        m_val = m_test = None
                        if self.cfg.base.mode == "train" or "val" in self.cfg.base.mode:
                            _val_report = val_report.split("\n")[i]
                            k_re = k.replace("(", "\(").replace(")", "\)")
                            m_val = re.match(rf".*{k_re} (\d+(?:\.\d+)?)", _val_report)
                            val_score = round(float(m_val.group(1)), v)
                        if self.cfg.base.mode == "train" or "test" in self.cfg.base.mode:
                            _test_report = test_report.split("\n")[i]
                            k_re = k.replace("(", "\(").replace(")", "\)")
                            m_test = re.match(rf".*{k_re} (\d+(?:\.\d+)?)", _test_report)
                            test_score = round(float(m_test.group(1)), v)
                        ret_keys += f"{k}, "
                        if m_val and m_test:
                            ret_values += f"{val_score} / {test_score}, "
                        elif m_val:
                            ret_values += f"{val_score}, "
                        elif m_test:
                            ret_values += f"{test_score}, "
                    save_log(ret_keys, self.log_path)
                    save_log(ret_values, self.log_path)
                save_log(f"results were saved to {self.log_dir}", self.log_path)

    def run(self, score_format=None):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.cfg.base.gpu_id)
        best_score = -float("inf")
        updated_epoch = 0
        if self.cfg.base.mode == "train":
            start_epoch = self.cfg.base.resume if self.cfg.base.resume else 1
            save_log(f"start training from epoch: {start_epoch}", self.log_path)
            for epoch in range(start_epoch, self.cfg.base.epoch+1):
                now = datetime.datetime.now().strftime("%y%m%d_%H-%M-%S")
                save_log(f"time: {now}", self.log_path)
                if self.cfg.base.adapt_mode != "none": # cross-domain
                    self.train_adapt(epoch)
                else:                                  # single-domain
                    self.train(epoch)
                
                if self.eval_scores: # eval per iteration
                    self.model.eval()
                    score_report, _ = self.evaluate(epoch, self.val_loader, self.model)
                    if self.val_score > best_score:
                        save_log("=" * 60, self.log_path)
                        save_log(score_report, self.log_path)
                        save_log("=" * 60, self.log_path)
                        if self.cfg.base.save_all_bests:
                            model_path = osp.join(self.model_save, f'net_epoch_{epoch:02d}.pth')
                            torch.save(self.model.state_dict(), model_path)
                        model_path = osp.join(self.model_save, 'net_latest.pth')
                        torch.save(self.model.state_dict(), model_path)

                        best_score = self.val_score
                        updated_epoch = epoch
                    if self.cfg.solver.early_stop:
                        if epoch - updated_epoch >= self.cfg.solver.early_stop:
                            save_log("early-stop of training", self.log_path)
                            break
                if self.cfg.base.debug: break
            save_log("training was done", self.log_path)

        save_log("\nbest results", self.log_path)
        save_log("=" * 60, self.log_path)
        self.final_test(updated_epoch, score_format)
        save_log("=" * 60, self.log_path)
