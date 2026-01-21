from utils.registry import Registry
import torch
import torch.nn as nn
from typing import Optional

LOSSES = Registry('losses')


def build_gt_offset(gt_points, H, W, radius, device, stride: int = 1):
    gt_offset = torch.zeros(2, H, W, device=device)
    gt_mask = torch.zeros(H, W, device=device)

    if isinstance(gt_points, list) and len(gt_points) > 0:
        if isinstance(gt_points[0], torch.Tensor):
            points = gt_points[0]
        else:
            points = gt_points
    else:
        points = gt_points

    if isinstance(points, torch.Tensor):
        points = points.to(device)

    for (x, y) in points:
        x = x.item() if isinstance(x, torch.Tensor) else x
        y = y.item() if isinstance(y, torch.Tensor) else y

        cx = x / stride
        cy = y / stride

        cx_int = int(cx)
        cy_int = int(cy)

        for iy in range(cy_int - radius, cy_int + radius + 1):
            for ix in range(cx_int - radius, cx_int + radius + 1):
                if ix < 0 or iy < 0 or ix >= W or iy >= H:
                    continue

                gt_offset[0, iy, ix] = cx - ix
                gt_offset[1, iy, ix] = cy - iy
                gt_mask[iy, ix] = 1.0

    return gt_offset, gt_mask


@LOSSES.register()
class FocalLoss(torch.nn.Module):

    def __init__(
            self,
            alpha: int = 2,
            beta: int = 4,
            reduction: str = 'sum',
            weights: Optional[torch.Tensor] = None,
            density_weight: Optional[str] = None,
            normalize: bool = False,
            eps: float = 1e-6,
            return_map: bool = False
            ) -> None:

        super().__init__()

        assert reduction in ['mean', 'sum'], \
            f'Reduction must be either \'mean\' or \'sum\', got {reduction}'

        self.alpha = alpha
        self.beta = beta
        self.reduction = reduction
        self.weights = weights
        self.density_weight = density_weight
        self.normalize = normalize
        self.eps = eps
        self.return_map = return_map

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        B, C, _, _ = target.shape

        if self.weights is not None:
            assert self.weights.shape[0] == C, \
                'Number of weights must match the number of channels, ' \
                f'got {C} channels and {self.weights.shape[0]} weights'

        pos_inds = target.eq(1).float()
        neg_inds = target.lt(1).float()

        # neg_weights = torch.pow(1 - target, self.beta)
        neg_weights = torch.pow(torch.clamp(1 - target, min=0), self.beta)

        loss = torch.zeros((B, C), device=output.device)

        # avoid NaN when net output is 1.0 or 0.0
        output = torch.clamp(output, min=self.eps, max=1 - self.eps)

        pos_loss = torch.log(output) * torch.pow(1 - output, self.alpha) * pos_inds
        neg_loss = torch.log(1 - output + self.eps) * torch.pow(output, self.alpha) * neg_weights * neg_inds

        num_pos = pos_inds.float().sum(3).sum(2)
        pos_loss = pos_loss.sum(3).sum(2) # (2, 2)
        neg_loss = neg_loss.sum(3).sum(2) # (2, 2)

        for b in range(B):
            for c in range(C):
                density = torch.tensor([1]).to(neg_loss.device)
                if self.density_weight == 'linear':
                    density = torch.clamp(num_pos[b][c], min=1.0)
                elif self.density_weight == 'squared':
                    density = torch.clamp(num_pos[b][c] ** 2, min=1.0)
                elif self.density_weight == 'cubic':
                    density = torch.clamp(num_pos[b][c] ** 3, min=1.0)

                if num_pos[b][c] == 0:
                    loss[b][c] = loss[b][c] - neg_loss[b][c]
                else:
                    loss[b][c] = density * (loss[b][c] - (pos_loss[b][c] + neg_loss[b][c]))
                    if self.normalize:
                        loss[b][c] = loss[b][c] / (num_pos[b][c] + self.eps)

        if self.weights is not None:
            loss = self.weights * loss

        if self.reduction == 'mean':

            return loss.mean()
        elif self.reduction == 'sum':

            return loss.sum()


# retinanet focal loss
class WithFocalLoss(nn.Module):
    """
        Focal loss for point / heatmap prediction
        outputs: sigmoid outputs in [0, 1]
        target: heatmap in [0, 1]
    """
    def __init__(self, alpha=2.0, beta=4.0, eps=1e-3):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(self, outputs: torch.Tensor, target: torch.Tensor):
        """
            outputs: (B, 1, H, W)  sigmoid output
            target: (B, 1, H, W)  heatmap [0,1]
        """
        outputs = torch.clamp(outputs, self.eps, 1.0 - self.eps)

        pos_inds = target.eq(1).float()
        neg_inds = target.lt(1).float()

        neg_weights = torch.pow(1 - target, self.beta)

        pos_loss = -torch.log(outputs) * torch.pow(1 - outputs, self.alpha) * pos_inds
        neg_loss = -torch.log(1 - outputs) * torch.pow(outputs, self.alpha) * neg_weights * neg_inds

        num_pos = pos_inds.sum()

        loss = pos_loss.sum() + neg_loss.sum()
        loss = loss / torch.clamp(num_pos, min=1.0) # loss.mean()

        return loss






