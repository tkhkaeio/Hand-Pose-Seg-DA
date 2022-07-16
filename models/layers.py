import torch.nn as nn

Pool = nn.MaxPool2d

class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=True)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU()
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Residual(nn.Module):
    def __init__(self, num_in, num_out, use_dropout=False):
        super(Residual, self).__init__()
        self.num_in = num_in
        self.num_out = num_out
        self.bn = nn.BatchNorm2d(self.num_in)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(self.num_in, self.num_out // 2, bias=True, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(self.num_out // 2)
        self.conv2 = nn.Conv2d(self.num_out // 2, self.num_out // 2, bias=True, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(self.num_out // 2)
        self.conv3 = nn.Conv2d(self.num_out // 2, self.num_out, bias=True, kernel_size=1)
        self.do = nn.Dropout(p=0.3)
        self.use_dropout = use_dropout

        if self.num_in != self.num_out:
            self.conv4 = nn.Conv2d(self.num_in, self.num_out, bias=True, kernel_size=1)

    def forward(self, x):
        residual = x
        out = self.bn(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        if self.use_dropout:
            out = self.do(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        if self.use_dropout:
            out = self.do(out)
        out = self.conv3(out)

        if self.num_in != self.num_out:
            residual = self.conv4(x)

        return out + residual

class Hourglass(nn.Module):
    def __init__(self, n, n_modules, feat_dim=256, use_dropout=False):
        super(Hourglass, self).__init__()
        self.n = n
        self.n_modules = n_modules
        self.feat_dim = feat_dim

        _up1_, _low1_, _low2_, _low3_ = [], [], [], []
        for j in range(self.n_modules):
            _up1_.append(Residual(self.feat_dim, self.feat_dim, use_dropout=use_dropout))
        self.low1 = nn.MaxPool2d(kernel_size=2, stride=2)
        for j in range(self.n_modules):
            _low1_.append(Residual(self.feat_dim, self.feat_dim, use_dropout=use_dropout))

        if self.n > 1:
            self.low2 = Hourglass(n - 1, self.n_modules, self.feat_dim, use_dropout=use_dropout)
        else:
            for j in range(self.n_modules):
                _low2_.append(Residual(self.feat_dim, self.feat_dim, use_dropout=use_dropout))
            self.low2_ = nn.ModuleList(_low2_)

        for j in range(self.n_modules):
            _low3_.append(Residual(self.feat_dim, self.feat_dim, use_dropout=use_dropout))

        self.up1_ = nn.ModuleList(_up1_)
        self.low1_ = nn.ModuleList(_low1_)
        self.low3_ = nn.ModuleList(_low3_)

        self.up2 = nn.Upsample(scale_factor=2)

    def forward(self, x):
        up1 = x
        for j in range(self.n_modules):
            up1 = self.up1_[j](up1)

        low1 = self.low1(x)
        for j in range(self.n_modules):
            low1 = self.low1_[j](low1)

        if self.n > 1:
            low2 = self.low2(low1)
        else:
            low2 = low1
            for j in range(self.n_modules):
                low2 = self.low2_[j](low2)

        low3 = low2
        for j in range(self.n_modules):
            low3 = self.low3_[j](low3)
        up2 = self.up2(low3)

        return up1 + up2