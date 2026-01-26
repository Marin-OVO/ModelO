"""
    train and val

        2026-01-26 experiment: unet -> dense map
        engine6th
"""
import argparse
import os
import pandas as pd

import torch
from torch.utils.data import DataLoader

from model import UNet
from datasets import CrowdDataset
from utils import custom_collate_fn
from utils.metrics import PointsMetrics
from utils.engine.engine6th import train_one_epoch, val_one_epoch
from utils.logger import setup_default_logging, time_str

import albumentations as A
from datasets.transforms import DownSample, DensityMap, FIDT, CustomTransformWrapper


# trainval
def args_parser():

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--data_root', default='data/crowdsat', type=str)

    # training parameters
    parser.add_argument('--epoch', default=150, type=int, metavar='N')
    parser.add_argument('--batch_size', default=16, type=int, metavar='N')
    parser.add_argument('--lr', default=0.0003, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--num_worker', default=6, type=int)

    parser.add_argument('--bilinear', default=True, type=bool)

    # device
    parser.add_argument('--device', default='cuda', type=str)

    # logging
    parser.add_argument('--output_path', default='weights', type=str)
    parser.add_argument('--save', default=None, type=str)
    parser.add_argument('--print_freq', default=8, type=int)

    # val/checkpoint strategy
    parser.add_argument('--checkpoint', default='best', type=str,
                        choices=['best', 'all', 'latest'])
    parser.add_argument('--select_mode', default='min', type=str,
                        choices=['min', 'max'])

    # dataset processing settings
    parser.add_argument('--radius', default=2, type=int)

    args = parser.parse_args()

    if args.save is None:
        args.save = os.path.join(args.output_path, 'best_model.pth')
    else:
        args.save = os.path.join(args.output_path, args.save)

    return args


def main(args):

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_path, exist_ok=True)

    # output with time
    timestr = time_str()
    work_dir = os.path.join(args.output_path, f'run_{timestr}'.format(timestr))
    os.makedirs(work_dir, exist_ok=True)

    logger_path = os.path.join(work_dir, 'log')
    os.makedirs(logger_path, exist_ok=True)

    logger, timestr = setup_default_logging('training', logger_path)

    # save csv
    csv_path = os.path.join(logger_path, 'training_log.csv')

    logger.info('=' * 60)
    logger.info('Training Configuration:')
    for arg, value in vars(args).items():
        logger.info(f'{arg}: {value}')

    model = UNet(num_ch=3, num_class=1, bilinear=args.bilinear)
    model.to(device)

    scaler = torch.cuda.amp.GradScaler()
    logger.info(f'Model created and moved to {device}')

    # optimizer -> AdamW
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    # lr_scheduler -> cosine annealing
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epoch,
        eta_min=1e-6
    )

    # 50% H/V flip + normalize + point -> mask + to_tensor
    train_albu_transforms = [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ]
    train_end_transforms = [
        CustomTransformWrapper(
            FIDT(radius=args.radius, num_classes=2, down_ratio=1),
            DensityMap(sigma=4.0)
        )
    ]

    # normalize + point -> mask + to_tensor
    val_albu_transforms = [
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225))
    ]
    val_end_transforms = [
        DownSample(down_ratio=1,
                   crowd_type='point')
    ]

    train_dataset = CrowdDataset(
        data_root=args.data_root,
        train=True,
        train_list="crowd_train.list",
        val_list="crowd_val.list",
        albu_transforms=train_albu_transforms,
        end_transforms=train_end_transforms
    )
    val_dataset = CrowdDataset(
        data_root=args.data_root,
        train=False,
        train_list="crowd_train.list",
        val_list="crowd_val.list",
        albu_transforms=val_albu_transforms,
        end_transforms=val_end_transforms
    )
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f'Train dataset size: {len(train_dataset)}')
    logger.info(f'Val dataset size: {len(val_dataset)}')
    logger.info(f'Total parameters: {total_params / 1e6:.2f} M')
    logger.info(f'Trainable parameters: {trainable_params / 1e6:.2f} M')

    train_dataloader = DataLoader(
        dataset = train_dataset,
        batch_size = args.batch_size,
        shuffle = True,
        pin_memory=True,
        num_workers=args.num_worker,
        collate_fn=custom_collate_fn
    )
    val_dataloader = DataLoader(
        dataset = val_dataset,
        batch_size = 1,
        shuffle = False,
        pin_memory=True,
        num_workers=args.num_worker,
    )

    checkpoints = args.checkpoint
    select = args.select_mode
    validate_on = 'mae'
    print_freq = args.print_freq

    assert checkpoints in ['best', 'all', 'latest']
    assert select in ['min', 'max']

    last_epoch = 0
    best_epoch = -1

    if select == 'min':
        best_val = float('inf')
    elif select == 'max':
        best_val = 0

    # training loop
    logger.info('Start Training:')

    train_loss_list = []
    lr_list = []
    mae_list = []
    rmse_list = []

    for epoch in range(last_epoch, args.epoch): # (0 - 149)
        logger.info('=' * 60)
        logger.info('Epoch [{:^3}/{:^3}]'.format(epoch + 1, args.epoch))

        # train
        loss, lr = train_one_epoch(
            model=model,
            train_dataloader=train_dataloader,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            device=device,
            logger=logger,
            print_freq=print_freq,
            args=args
        )
        train_loss_list.append(loss)
        lr_list.append(lr)

        # val
        tmp_results = val_one_epoch(
            model=model,
            val_dataloader=val_dataloader,
            epoch=epoch,
            args=args
        )
        mae_list.append(tmp_results["mae"])
        rmse_list.append(tmp_results["rmse"])

        is_best = False

        if select == 'min':
            if tmp_results[validate_on] < best_val:
                best_val = tmp_results[validate_on]
                best_epoch = epoch # not tmp epoch
                is_best = True

        elif select == 'max':
            if tmp_results[validate_on] > best_val:
                best_val = tmp_results[validate_on]
                best_epoch = epoch
                is_best = True

        # Save checkpoints
        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': lr_scheduler.state_dict(),
            'loss': loss,
            'best_val': best_val,
            'config': vars(args)
        }

        if checkpoints == 'best':
            if is_best:
                outpath = os.path.join(work_dir, 'best_model.pth')
                torch.save(state, outpath)
                logger.info(f'Saved best model to {outpath}')
        elif checkpoints == 'latest':
            outpath = os.path.join(work_dir, 'latest_model.pth')
            torch.save(state, outpath)
        elif checkpoints == 'all':
            if is_best:
                outpath = os.path.join(work_dir, 'best_model.pth')
                torch.save(state, outpath)
            outpath = os.path.join(work_dir, 'latest_model.pth')
            torch.save(state, outpath)

        # add the best info to results
        tmp_results['best_val'] = best_val
        tmp_results['best_epoch'] = best_epoch + 1
        tmp_results['train_loss'] = loss

        # log to csv
        data_frame = pd.DataFrame([tmp_results])
        if epoch == 0:
            data_frame.to_csv(csv_path, mode='w', header=True, index=False)
        else:
            data_frame.to_csv(csv_path, mode='a', header=False, index=False)

        # log val results
        logger.info(
            f"Val Results: "
            f"Epoch: [{epoch + 1:^3}/{args.epoch:^3}] | "
            f"MAE: {tmp_results['mae']:^8.4f} | "
            f"RMSE: {tmp_results['rmse']:^8.4f} | "
            f"Best-Val({validate_on}): {best_val:^8.4f} | "
            f"Best-Epoch: {best_epoch + 1:^3} "
        )

        lr_scheduler.step()

    logger.info('=' * 60)
    logger.info('Training Complete!')
    logger.info(f'Best {validate_on}: {best_val:.4f} at epoch {best_epoch + 1}')
    logger.info(f'Results saved to: {work_dir}')
    logger.info('=' * 60)

    if len(train_loss_list) != 0 and len(lr_list) != 0:
        from utils.plot_curve import plot_loss_and_lr
        plot_loss_and_lr(train_loss_list, lr_list, work_dir)

    if len(mae_list) != 0:
        from utils.plot_curve import plot_mae
        plot_mae(mae_list, work_dir)

    if len(rmse_list) != 0:
        from utils.plot_curve import plot_rmse
        plot_rmse(rmse_list, work_dir)

    if len(tmp_results['gt']) != 0 and len(tmp_results['dense']) != 0:
        from utils.plot_curve import plot_dense
        plot_dense(tmp_results['gt'], tmp_results['dense'], work_dir)


if __name__ == "__main__":
    args = args_parser()

    main(args)