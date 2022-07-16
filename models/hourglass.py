import torch
import torch.nn as nn
import torch.nn.functional as F
from helpers.util import compute_uv_from_heatmaps
from models.layers import Hourglass, Residual
Pool = nn.MaxPool2d

def get_hourglass_net(model_type, num_classes, num_stages=2, num_modules=2, in_channels=3, use_dropout=False, feat_size=64):
    # print(model_type)
    if model_type=="hourglass_2d": #pose only
        model = HeatmapHourglass(num_classes, in_channels, num_stages, num_modules, feat_size, use_dropout)
    elif model_type=="hourglass_joint_2d": #pose & seg for 2d rgb
        model = HeatmapHourglassSeg(num_classes, in_channels, num_stages, num_modules, feat_size, use_dropout)
    else:
        print("model does not exist", model_type)
        raise NotImplementedError()
    
    return model

class HeatmapHourglass(nn.Module):
    def __init__(self, num_joints, num_stages=2, num_modules=2, feat_size=64, use_dropout=False):
        super(HeatmapHourglass, self).__init__()
        self.num_joints = num_joints
        self.num_output = num_joints
        self.n_stack = num_stages

        self.n_modules = num_modules
        self.feat_size = feat_size
        self.feat_downsample = 2
        self.feat_dim = 256

        self.conv1_ = nn.Conv2d(3, 64, bias=True, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.r1 = Residual(64, 128, use_dropout=use_dropout)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.r4 = Residual(128, 128, use_dropout=use_dropout)
        self.r5 = Residual(128, self.feat_dim, use_dropout=use_dropout)
        self.do = nn.Dropout(p=0.3)
        self.use_dropout = use_dropout

        _hourglass, _residual, _lin_, _tmp_out, _ll_, _tmp_out_ = [], [], [], [], [], []
        for i in range(self.n_stack):
            _hourglass.append(Hourglass(4, self.n_modules, self.feat_dim, use_dropout=use_dropout))
            for j in range(self.n_modules):
                _residual.append(Residual(self.feat_dim, self.feat_dim, use_dropout=use_dropout))
            lin = nn.Sequential(nn.Conv2d(self.feat_dim, self.feat_dim, bias=True, kernel_size=1, stride=1),
                                nn.BatchNorm2d(self.feat_dim), self.relu)
            _lin_.append(lin)
            _tmp_out.append(nn.Conv2d(self.feat_dim, self.num_output, bias=True, kernel_size=1, stride=1))
            if i < self.n_stack - 1:
                _ll_.append(nn.Conv2d(self.feat_dim, self.feat_dim, bias=True, kernel_size=1, stride=1))
                _tmp_out_.append(nn.Conv2d(self.num_output, self.feat_dim, bias=True, kernel_size=1, stride=1))

        self.hourglass = nn.ModuleList(_hourglass)
        self.residual = nn.ModuleList(_residual)
        self.lin_ = nn.ModuleList(_lin_)
        self.tmp_out = nn.ModuleList(_tmp_out)
        self.ll_ = nn.ModuleList(_ll_)
        self.tmp_out_ = nn.ModuleList(_tmp_out_)

    def forward(self, x, with_feat=False):
        x = self.conv1_(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.use_dropout:
            x = self.do(x)
        x = self.r1(x)
        if self.use_dropout:
            x = self.do(x)
        # x = self.maxpool(x)
        x = self.r4(x)
        if self.use_dropout:
            x = self.do(x)
        x = self.r5(x)
        # print(x.shape) torch.Size([64, 256, 64, 64])

        out = []
        encoding = []

        for i in range(self.n_stack):
            hg = self.hourglass[i](x)
            if self.use_dropout:
                hg = self.do(hg)
            ll = hg
            for j in range(self.n_modules):
                ll = self.residual[i * self.n_modules + j](ll)
                if self.use_dropout:
                    ll = self.do(ll)
            ll = self.lin_[i](ll)
            tmp_out = self.tmp_out[i](ll)
            out.append(tmp_out)
            if i < self.n_stack - 1:
                ll_ = self.ll_[i](ll)
                tmp_out_ = self.tmp_out_[i](tmp_out)
                x = x + ll_ + tmp_out_
                encoding.append(x)
            else:
                encoding.append(ll)

        hm = out[-1]
        jt_uv = compute_uv_from_heatmaps(hm, [self.feat_size * self.feat_downsample, self.feat_size * self.feat_downsample])
        
        if with_feat:
            return jt_uv, hm, None, torch.stack(out), torch.stack(encoding)
        else:
            return jt_uv, hm, None

class HeatmapHourglassSeg(nn.Module):
    def __init__(self, num_joints, inchannels=3, num_stages=2, num_modules=2, feat_size=64, use_dropout=False):
        super(HeatmapHourglassSeg, self).__init__()
        self.dummy_param = nn.Parameter(torch.empty(0))
        self.num_joints = num_joints
        self.num_output = num_joints
        self.inchannels = inchannels
        self.n_stack = num_stages

        self.n_modules = num_modules
        self.feat_size = feat_size
        self.feat_downsample = 2
        self.feat_dim = 256
        self.kernel_size = 1

        self.conv1_ = nn.Conv2d(self.inchannels, 64, bias=True, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.r1 = Residual(64, 128, use_dropout=use_dropout)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.r4 = Residual(128, 128, use_dropout=use_dropout)
        self.r5 = Residual(128, self.feat_dim, use_dropout=use_dropout)
        self.do = nn.Dropout(p=0.3)
        self.use_dropout = use_dropout

        deconv_num = 1 #4 - int(math.log(self.feat_downsample, 2))
        deconv_kernel = [4] * deconv_num
        deconv_planes = [256] * deconv_num
        self.deconv_with_bias = False
        self.deconv_layers = self._make_deconv_layer(deconv_num, deconv_kernel, deconv_planes, use_dropout=use_dropout)

        _hourglass, _residual, _lin_, _tmp_out, _ll_, _tmp_out_ = [], [], [], [], [], []
        for i in range(self.n_stack):
            _hourglass.append(Hourglass(4, self.n_modules, self.feat_dim, use_dropout=use_dropout))
            _residual.append(Residual(self.feat_dim, self.feat_dim, use_dropout=use_dropout))
            for j in range(self.n_modules-1):
                _residual.append(Residual(self.feat_dim, self.feat_dim, use_dropout=use_dropout))
            lin = nn.Sequential(nn.Conv2d(self.feat_dim, self.feat_dim, bias=True, kernel_size=1, stride=1),
                                nn.BatchNorm2d(self.feat_dim), self.relu)
            _lin_.append(lin)
            if self.inchannels == 1:
                _tmp_out.append(nn.Conv2d(self.feat_dim , self.num_output * 3, bias=True, kernel_size=1, stride=1))
            else:
                _tmp_out.append(nn.Conv2d(self.feat_dim, self.num_output, bias=True, kernel_size=1, stride=1))
            # _tmp_out.append(nn.Conv2d(self.feat_dim, 3*self.num_output, bias=True, kernel_size=1, stride=1))
            if i < self.n_stack - 1:
                _ll_.append(nn.Conv2d(self.feat_dim, self.feat_dim, bias=True, kernel_size=1, stride=1))
                if self.inchannels == 1:
                    _tmp_out_.append(nn.Conv2d(self.num_output * 3, self.feat_dim, bias=True, kernel_size=1, stride=1))
                else:
                    _tmp_out_.append(nn.Conv2d(self.num_output, self.feat_dim, bias=True, kernel_size=1, stride=1))
                # _tmp_out_.append(nn.Conv2d(self.num_output, self.feat_dim, bias=True, kernel_size=1, stride=1))
        
        self.mask_head = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, stride=1, padding=0)
       
        self.hourglass = nn.ModuleList(_hourglass)
        self.residual = nn.ModuleList(_residual)
        self.lin_ = nn.ModuleList(_lin_)
        self.tmp_out = nn.ModuleList(_tmp_out)
        self.ll_ = nn.ModuleList(_ll_)
        self.tmp_out_ = nn.ModuleList(_tmp_out_)

    def _make_deconv_layer(self, num_layers, kernels, planes, use_dropout=False):

        layers = []
        self.inplanes = 256
        for i in range(num_layers):
            kernel, padding, output_padding = kernels[i], 1, 0
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes[i],
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes[i], momentum=0.1))
            layers.append(nn.ReLU(inplace=True))
            if use_dropout and i != (num_layers - 1):
                layers.append(nn.Dropout(p=0.3))
            self.inplanes = planes[i]

        return nn.Sequential(*layers)

    def forward(self, x, with_feat=False, with_enc_feat=False, freeze_mask_head=False, soft_argmax=False):
        x = self.conv1_(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.use_dropout:
            x = self.do(x)
        x = self.r1(x)
        if self.use_dropout:
            x = self.do(x)
        # x = self.maxpool(x)
        x = self.r4(x)
        if self.use_dropout:
            x = self.do(x)
        x = self.r5(x)
        
        x = self.deconv_layers(x)
        # print(x.shape) #torch.Size([1, 256, 128, 128])

        if freeze_mask_head:
            with torch.no_grad():
                mask = torch.sigmoid(self.mask_head(x))
        else:
            mask = torch.sigmoid(self.mask_head(x))
        out = []
        encoding = []

        x = F.interpolate(x, [self.feat_size, self.feat_size]) #cat_x
        for i in range(self.n_stack):
            hg = self.hourglass[i](x)
            if self.use_dropout:
                hg = self.do(hg)
            ll = hg
            for j in range(self.n_modules):
                ll = self.residual[i * self.n_modules + j](ll)
                if self.use_dropout:
                    ll = self.do(ll)
            ll = self.lin_[i](ll)
            tmp_out = self.tmp_out[i](ll)
            out.append(tmp_out)
            if i < self.n_stack - 1:
                ll_ = self.ll_[i](ll)
                tmp_out_ = self.tmp_out_[i](tmp_out)
                x = x + ll_ + tmp_out_
                encoding.append(x)
            else:
                encoding.append(ll)
        
        hm = out[-1]
        jt_uvd = compute_uv_from_heatmaps(hm, [self.feat_size * self.feat_downsample, self.feat_size * self.feat_downsample], soft_argmax=soft_argmax)
        offset = hm

        if with_feat:
            return jt_uvd, offset, mask, torch.stack(encoding)[-1]
        elif with_enc_feat:
            return jt_uvd, offset, mask, torch.stack(out), torch.stack(encoding)
        else:
            return jt_uvd, offset, mask