import torch
import time
from torch.utils.data import DataLoader
from easydict import EasyDict
from tensorboardX import SummaryWriter

from utils import AverageMeter, save_model
from model import CascadeRCNN
from config import cfg
from dataset import WSGISDDataset, cls2idx

writer = SummaryWriter()
global_step = 0


def train_one_epoch(model, dl, optim):
    global global_step
    summary_loss = AverageMeter()

    t = time.time()
    model.train()
    for step, batched_inputs in enumerate(dl):
        result = model(batched_inputs)
        losses = []

        lr = optim.param_groups[0]['lr']
        writer.add_scalar('lr', lr, global_step=global_step)
        step_log = f'Train Step {step}/{len(dl)}, LR: {lr}, '
        for k, v in result.items():
            writer.add_scalar(k, v, global_step=global_step)
            losses.append(v)
            if k == 'loss_mask':
                step_log += f'{k}: {v:.3f}, '

        step_log += f'summary_loss: {summary_loss.avg:.3f}, '
        step_log += f'time: {(time.time() - t):.3f}'
        print(step_log, end='\r')

        loss = sum(losses)
        optim.zero_grad()
        loss.backward()
        optim.step()

        summary_loss.update(loss.detach().item(), len(batched_inputs))
        writer.add_scalar('summary_loss', summary_loss.avg, global_step=global_step)
        global_step += 1
    return summary_loss


def val_one_epoch(model, dl):
    summary_loss = AverageMeter()

    t = time.time()
    model.eval()
    with torch.no_grad():
        for step, batched_inputs in enumerate(dl):
            result = model(batched_inputs)
            losses = []

            step_log = f'Train Step {step}/{len(dl)}, '
            for k, v in result.items():
                losses.append(v)
                if k == 'loss_mask':
                    step_log += f'{k}: {v:.3f}, '

            step_log += f'summary_loss: {summary_loss.avg:.3f}, '
            step_log += f'time: {(time.time() - t):.3f}'
            print(step_log, end='\r')

            loss = sum(losses)
            summary_loss.update(loss.detach().item(), len(batched_inputs))
        return summary_loss


def train(cfg):
    ds = WSGISDDataset(root='datasets/wgisd', resize=cfg.RESIZE)
    dl = torch.utils.data.DataLoader(ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
                                     collate_fn=ds.collate_fn, num_workers=cfg.NUM_WORKERS)

    val_ds = WSGISDDataset(root='datasets/wgisd', resize=cfg.RESIZE, mode='test')
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=True,
                                         collate_fn=ds.collate_fn, num_workers=4)

    model = CascadeRCNN(cfg).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, threshold=cfg.lr*0.1, cooldown=1)
    best_summary_loss = 1e6

    for epoch in range(cfg.EPOCH):
        t = time.time()
        summary_loss = train_one_epoch(model, dl, optimizer)
        info = f'[RESULT]: Train. Epoch: {epoch:02d}, summary_loss: {summary_loss.avg:.3f}, time: {(time.time() - t):.3f}, '
        print(info)

        summary_loss = val_one_epoch(model, val_dl)
        info = f'[EVAL]: Train. Epoch: {epoch:02d}, summary_loss: {summary_loss.avg:.3f}, time: {(time.time() - t):.3f}, '
        print(info)
        
        scheduler.step(summary_loss.avg)
        if summary_loss.avg < best_summary_loss:
            best_summary_loss = summary_loss.avg
            save_model(cfg, model, optimizer, best_summary_loss, epoch, f'best-checkpoint.bin')
        save_model(cfg, model, optimizer, best_summary_loss, epoch, f'last-checkpoint.bin')


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    cfg.NUM_CLASSES = len(cls2idx)
    train(cfg)
