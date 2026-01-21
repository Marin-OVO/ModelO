"""
    2026-01-18 experiment: count then loc
"""
import time
from typing import Optional, Union

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.lmds import LMDS
from utils.logger import *
from utils.averager import *
from utils.loss import *

from torch.cuda.amp import autocast, GradScaler


def train_one_epoch(
        model: torch.nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        scaler,
        epoch: int,
        device: torch.device,
        logger: logging.Logger,
        print_freq: int,
        args,
        ) -> float:
    # init
    first_order_loss = AverageMeter(20)
    second_order_loss = AverageMeter(20)
    third_order_loss = AverageMeter(20)
    forth_order_loss = AverageMeter(20)
    fifth_order_loss = AverageMeter(20)
    losses = AverageMeter(20)
    batch_times = AverageMeter(20)

    freq = len(train_dataloader) // print_freq
    print_freq_lst = [i * freq for i in range(1, 8)]
    print_freq_lst.append(len(train_dataloader) - 1)

    batch_start = time.time()

    model.train()
    lr = optimizer.param_groups[0]['lr']

    # lmds
    ks = args.lmds_kernel_size
    if isinstance(ks, int):
        ks = (ks, ks)
    lmds = LMDS(
        kernel_size=ks,
        adapt_ts=args.lmds_adapt_ts
    )

    # FocalLoss
    FL = FocalLoss(reduction='mean', normalize=True)
    # FL = WithFocalLoss(alpha=2.0, beta=4.0)
    for step, (images, targets) in enumerate(train_dataloader):

        # img
        images = images.to(device)

        with (autocast()):
            # train outputs
            outputs = model(images)  # dict
            heatmap_out = outputs['heatmap_out']
            offset_out = outputs['offset_out']
            density_out = outputs['density_out']
            # heatmap_out_freq = outputs['heatmap_out_freq']
            # offset_out_freq = outputs['offset_out_freq']
            # density_out_freq = outputs['density_out_freq']

            # mask
            _, _, H, W = offset_out.shape
            gt_heatmap = targets['fidt_map'].to(device)
            gt_dense_map = targets['density_map'].to(device)
            gt_points = targets['points']
            gt_count = targets['points'].shape[1]

            gt_offset, gt_mask = build_gt_offset(
                gt_points, H, W,
                radius=args.radius,
                device=args.device,
                stride=1
            )
            gt_offset = gt_offset.to(device)
            gt_mask = gt_mask.to(device)

            gt_offset = gt_offset.unsqueeze(0)
            gt_mask = gt_mask.unsqueeze(0).unsqueeze(0)
            batch_size = offset_out.shape[0]
            gt_offset = gt_offset.expand(batch_size, -1, -1, -1)
            gt_mask = gt_mask.expand(batch_size, -1, -1, -1)

            counts, _, _, _ = lmds(outputs['heatmap_out'])
            counts = torch.tensor(
                counts[0][0],
                device=density_out.device,
                dtype=density_out.dtype
            )

            # loss
            heatmap_loss = FL(heatmap_out, gt_heatmap)
            # heatmap_loss_freq = FL(heatmap_out_freq, gt_heatmap)

            offset_loss = (F.smooth_l1_loss(offset_out, gt_offset, reduction='none') * gt_mask).sum() / (gt_mask.sum() + 1e-6)
            density_loss = F.mse_loss(density_out, gt_dense_map)
            cons_loss = F.l1_loss(density_out.sum(), gt_count)

            heatmap_weight = 1.0
            offset_weight = 1.0
            density_weight = 0.1
            cons_weight = 1.0
            # offset_weight = 1.0 if epoch >= args.unfreeze[0] else 0.0
            # density_weight = 1.0 if epoch >= args.unfreeze[1] else 0.0
            # cons_weight = 0.001 if epoch >= args.unfreeze[1] else 0.0

            total_loss = ((heatmap_weight * heatmap_loss +
                          offset_weight * offset_loss +
                          density_weight * density_loss +
                          cons_weight * cons_loss))
                          # 0.03 * (heatmap_loss_freq))

        first_order_loss.update(heatmap_loss.detach().cpu().item())
        second_order_loss.update(offset_loss.detach().cpu().item())
        third_order_loss.update(density_loss.detach().cpu().item())
        forth_order_loss.update(cons_loss.detach().cpu().item())
        fifth_order_loss.update(0.0)
        losses.update(total_loss.detach().cpu().item())

        optimizer.zero_grad()
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if step in print_freq_lst:
            logger.info(
                "Epoch [{:^3}/{:^3}] | Iter {:^5} | LR {:.6f} | "
                "First {:.3f}({:.3f}) | "
                "Second {:.3f}({:.3f}) | "
                "Third {:.3f}({:.3f}) | "
                "Forth {:.3f}({:.3f}) | "
                "Total {:.3f}({:.3f})".format(
                    epoch + 1, args.epoch,
                    step, lr,
                    first_order_loss.val, first_order_loss.avg,
                    second_order_loss.val, second_order_loss.avg,
                    third_order_loss.val, third_order_loss.avg,
                    forth_order_loss.val, forth_order_loss.avg,
                    losses.val, losses.avg,
                )
            )

    out = losses.avg

    batch_end = time.time()
    batch_times.update(batch_end-batch_start)

    logger.info(
        "Epoch [{:^3}/{:^3}] | LR {:.6f} | "
        "Total {:.3f}({:.3f}) | "
        "Time {:.2f}s".format(
            epoch + 1, args.epoch, lr,
            losses.val, losses.avg,
            batch_times.avg,
        )
    )

    return out, lr


@torch.no_grad
def val_one_epoch(
        model: torch.nn.Module,
        val_dataloader: torch.utils.data.DataLoader,
        epoch: int,
        metrics: object,
        args
        ) -> Union[float, torch.Tensor]:

    metrics.flush()
    iter_metrics = metrics.copy()

    model.eval()

    for step, (images, targets) in enumerate(val_dataloader):
        images = images.cuda()

        # val outputs
        outputs = model(images)

        points = targets['points']
        labels = targets['labels']

        if isinstance(labels, torch.Tensor):
            labels = labels.squeeze(0).tolist()

        points = np.asarray(points) # (1, N, 2)

        # (N, 2, 1) -> (N, 2)
        if points.ndim == 3 and points.shape[-1] == 1:
            points = points.squeeze(-1)

        # downsample (1, N, 2) -> (N, 2)
        if points.ndim == 3 and points.shape[0] == 1:
                points = points.squeeze(0)

        assert points.ndim == 2 and points.shape[1] == 2, \
            f"Invalid GT points shape: {points.shape}"

        gt_coords = [(int(p[1]), int(p[0])) for p in points]
        gt_labels = labels

        gt = dict(
            loc=gt_coords,
            labels=gt_labels
        )

        # int -> (tuple)
        ks = args.lmds_kernel_size
        if isinstance(ks, int):
            ks = (ks, ks)

        lmds = LMDS(
            kernel_size=ks,
            adapt_ts=args.lmds_adapt_ts
        )

        counts, locs, labels, scores = lmds(outputs['heatmap_out'])

        locs_pred = np.asarray(locs[0])
        if locs_pred.ndim == 3 and locs_pred.shape[-1] == 1:
            locs_pred = locs_pred.squeeze(-1)

        preds = dict(
            loc=locs_pred,
            labels=labels[0],
            scores=scores[0],
        )

        iter_metrics.feed(**dict(gt=gt, preds=preds))
        iter_metrics.aggregate()

        iter_metrics.flush()
        metrics.feed(**dict(gt=gt, preds=preds))

    mAP = np.mean([metrics.ap(c) for c in range(1, metrics.num_classes)]).item()

    metrics.aggregate()

    recall = metrics.recall()
    precision = metrics.precision()
    f1_score = metrics.fbeta_score()
    accuracy = metrics.accuracy()

    tmp_results = {
        'epoch': epoch + 1,
        'f1_score': f1_score,
        'recall': recall,
        'precision': precision,
        'accuracy': accuracy,
        "mAP": mAP
    }

    return tmp_results # a dict?