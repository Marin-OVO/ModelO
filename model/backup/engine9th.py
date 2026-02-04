"""
    2026-02-02 experiment: mse
    2026-02-03 experiment: add multi
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
    losses = AverageMeter(20)
    batch_times = AverageMeter(20)

    freq = len(train_dataloader) // print_freq
    print_freq_lst = [i * freq for i in range(1, 8)]
    print_freq_lst.append(len(train_dataloader) - 1)

    batch_start = time.time()

    model.train()
    lr = optimizer.param_groups[0]['lr']

    # FocalLoss
    FL = FocalLoss(reduction='mean', normalize=True)
    for step, (images, targets) in enumerate(train_dataloader):

        # img
        images = images.to(device)

        with ((autocast())):

            # train outputs
            gt_heatmap = targets['fidt_map'].to(device)
            gt_densitymap = targets['density_map'].to(device)
            outputs = model(images, gt_densitymap) # use mask multi
            heatmap_out = outputs['heatmap_out']
            x3 = outputs['x3_map_out']
            x5 = outputs['x5_map_out']

            # loss
            heatmap_loss = F.mse_loss(heatmap_out[:, 0:1, :, :], gt_heatmap)
            align_loss = F.mse_loss(x3, x5)

            heatmap_weight, align_weight = 1.0, 0.0
            total_loss = ((heatmap_weight * heatmap_loss) +
                          (align_weight * align_loss))


        first_order_loss.update(heatmap_loss.detach().cpu().item())
        second_order_loss.update(align_loss.detach().cpu().item())
        losses.update(total_loss.detach().cpu().item())

        optimizer.zero_grad()
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if step in print_freq_lst:
            logger.info(
                "Epoch [{:^3}/{:^3}] | Iter {:^5} | LR {:.6f} | "
                "First {:.3f}({:.3f}) | "
                # "Second {:.3f}({:.3f}) | "
                "Total {:.3f}({:.3f})".format(
                    epoch + 1, args.epoch,
                    step, lr,
                    first_order_loss.val, first_order_loss.avg,
                    # second_order_loss.val, second_order_loss.avg,
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

    gt_all = []
    dense_all = []

    for step, (images, targets) in enumerate(val_dataloader):
        images = images.cuda()

        # val outputs
        outputs = model(images)

        points = targets['points']
        labels = targets['labels']

        gt_points = targets['points']
        gt_counts = torch.tensor(
            [len(p) for p in gt_points],
            dtype=torch.float32,
            device=args.device
        )
        # gt_all.append(gt_counts.item())
        # dense_all.append(outputs['density_out'].sum(dim=(1, 2, 3)).item())

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

    gt_all = np.array(gt_all)
    dense_all = np.array(dense_all)

    tmp_results = {
        'epoch': epoch + 1,
        'f1_score': f1_score,
        'recall': recall,
        'precision': precision,
        'accuracy': accuracy,
        "mAP": mAP,
        'gt': gt_all,
        'dense': dense_all
    }

    return tmp_results