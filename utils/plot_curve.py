import os
import numpy as np

import datetime
import matplotlib.pyplot as plt


def plot_loss_and_lr(train_loss, learning_rate,  save_dir='./'):
    try:
        x = list(range(1, len(train_loss) + 1))
        fig, ax1 = plt.subplots(1, 1)
        ax1.plot(x, train_loss, 'r', label='loss')
        ax1.set_xlabel("epoch")
        ax1.set_ylabel("loss")
        ax1.set_title("Train Loss and lr")
        plt.legend(loc='best')

        ax2 = ax1.twinx()
        ax2.plot(x, learning_rate, label='lr')
        ax2.set_ylabel("learning rate")
        ax2.set_xlim(1, len(train_loss) + 1)  # 设置横坐标整数间隔
        plt.legend(loc='best')

        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        plt.legend(handles1 + handles2, labels1 + labels2, loc='upper right')

        fig.subplots_adjust(right=0.8)
        filename = os.path.join(save_dir, 'loss_and_lr.png')
        fig.savefig(filename)

        plt.close()
        print("successful save loss curve! ")
    except Exception as e:
        print(e)


def plot_map(mAP, save_dir='./'):
    try:
        x = list(range(1, len(mAP) + 1))
        plt.plot(x, mAP, label='mAp')
        plt.xlabel('epoch')
        plt.ylabel('mAP')
        plt.title('Eval mAP')
        plt.xlim(1, len(mAP) + 1)
        plt.legend(loc='best')

        filename = os.path.join(save_dir, 'mAP.png')
        plt.savefig(filename)
        plt.close()
        print("successful save mAP curve!")
    except Exception as e:
        print(e)

def plot_f1(f1, save_dir='./'):
    try:
        x = list(range(1, len(f1) + 1))
        plt.plot(x, f1, label='F1-score')
        plt.xlabel('epoch')
        plt.ylabel('F1-score')
        plt.title('Eval F1-score')
        plt.xlim(1, len(f1) + 1)
        plt.legend(loc='best')

        filename = os.path.join(save_dir, 'F1-score.png')
        plt.savefig(filename)
        plt.close()
        print("successful save F1-score curve!")
    except Exception as e:
        print(e)

def plot_mae(mae, save_dir='./'):
    try:
        x = list(range(1, len(mae) + 1))
        plt.plot(x, mae, label='MAE')
        plt.xlabel('epoch')
        plt.ylabel('MAE')
        plt.title('Eval MAE')
        plt.xlim(1, len(mae) + 1)
        plt.legend(loc='best')

        filename = os.path.join(save_dir, 'MAE.png')
        plt.savefig(filename)
        plt.close()
        print("successful save MAE curve!")
    except Exception as e:
        print(e)

def plot_rmse(rmse, save_dir='./'):
    try:
        x = list(range(1, len(rmse) + 1))
        plt.plot(x, rmse, label='RMSE')
        plt.xlabel('epoch')
        plt.ylabel('RMSE')
        plt.title('Eval RMSE')
        plt.xlim(1, len(rmse) + 1)
        plt.legend(loc='best')

        filename = os.path.join(save_dir, 'RMSE.png')
        plt.savefig(filename)
        plt.close()
        print("successful save RMSE curve!")
    except Exception as e:
        print(e)

def plot_dense(gt, pred, save_dir='./', name='dense'):
    gt = np.asarray(gt)
    pred = np.asarray(pred)

    max_val = max(gt.max(), pred.max())

    plt.figure(figsize=(6, 6))
    plt.scatter(gt, pred, s=10, alpha=0.6)
    plt.plot([0, max_val], [0, max_val], 'r--', linewidth=1)

    plt.xlabel('GT Count')
    plt.ylabel('Predicted Count')

    plt.xlim(0, max_val)
    plt.ylim(0, max_val)
    plt.grid(True, linestyle='--', alpha=0.5)

    save_path = os.path.join(save_dir, f'{name}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f'successful save {name}!')
