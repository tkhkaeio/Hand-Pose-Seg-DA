import torch

class SmoothL1Loss(torch.nn.Module):

    def __init__(self):
        super(SmoothL1Loss, self).__init__()

    def forward(self, x, y, mask=None, weight=None, reduce=True):
        total_loss = 0
        assert(x.shape == y.shape)
        z = (x - y).float()
        if mask is not None:
            z = z.view(z.size(0), z.size(1), -1)
            mask = mask.unsqueeze(2).repeat(1, int(z.size(1) / mask.size(1)), 1)
            z = z * mask
        mse_mask = (torch.abs(z) < 0.01).float()
        l1_mask = (torch.abs(z) >= 0.01).float()
        mse = mse_mask * z
        l1 = l1_mask * z
        if reduce:
            total_loss += torch.mean(self._calculate_MSE(mse)*mse_mask)
            total_loss += torch.mean(self._calculate_L1(l1)*l1_mask)
        else:
            total_loss += self._calculate_MSE(mse)*mse_mask
            total_loss += self._calculate_L1(l1)*l1_mask

        return total_loss

    def _calculate_MSE(self, z):
        return 0.5 *(torch.pow(z,2))

    def _calculate_L1(self,z):
        return 0.01 * (torch.abs(z) - 0.005)