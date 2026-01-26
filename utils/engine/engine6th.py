"""
    2026-01-26 experiment: unet -> dense map
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

from torch.cuda.amp import autocast


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
    losses = AverageMeter(20)
    batch_times = AverageMeter(20)

    freq = len(train_dataloader) // print_freq
    print_freq_lst = [i * freq for i in range(1, 8)]
    print_freq_lst.append(len(train_dataloader) - 1)

    batch_start = time.time()

    model.train()
    lr = optimizer.param_groups[0]['lr']

    for step, (images, targets) in enumerate(train_dataloader):

        # img
        images = images.to(device)
        with (autocast()):

            # train outputs
            outputs = model(images)
            gt_dense_map = targets['density_map'].to(device)

            # loss
            dense_loss = F.mse_loss(outputs, gt_dense_map, reduction='mean')
            cons_loss = F.l1_loss(outputs.sum(dim=(1, 2, 3)), gt_dense_map.sum(dim=(1, 2, 3)))

            dense_weight = 1.0
            cons_weight = 0.1

            total_loss = dense_loss * dense_weight + cons_loss * cons_weight

        first_order_loss.update(dense_loss.detach().cpu().item())
        second_order_loss.update(cons_loss.detach().cpu().item())
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
                "Total {:.3f}({:.3f})".format(
                    epoch + 1, args.epoch,
                    step, lr,
                    first_order_loss.val, first_order_loss.avg,
                    second_order_loss.val, second_order_loss.avg,
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
        args
        ) -> Union[float, torch.Tensor]:

    model.eval()
    mae_meter = AverageMeter(20)
    rmse_meter = AverageMeter(20)

    gt_all = []
    dense_all = []

    for step, (images, targets) in enumerate(val_dataloader):
        images = images.cuda()

        # val outputs
        outputs = model(images)

        gt_points = targets['points']
        gt_counts = torch.tensor(
            [len(p) for p in gt_points],
            dtype=torch.float32,
            device=args.device
        )

        gt_all.append(gt_counts.item())
        dense_all.append(outputs.sum(dim=(1, 2, 3)).item())

        diff = outputs.sum(dim=(1, 2, 3)) - gt_counts
        mae = torch.abs(diff).mean()
        rmse = torch.sqrt((diff ** 2).mean())

        mae_meter.update(mae.item())
        rmse_meter.update(rmse.item())

    gt_all = np.array(gt_all)
    dense_all = np.array(dense_all)

    tmp_results = {
        'epoch': epoch + 1,
        'mae': mae_meter.avg,
        'rmse': rmse_meter.avg,
        'gt': gt_all,
        'dense': dense_all
    }

    return tmp_results