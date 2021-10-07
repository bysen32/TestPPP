import argparse
import os
import time
import logging
import json
import warnings
from tqdm import tqdm
import numpy as np

from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import get_trainval_dataset
from utils.utils import AverageMeter, TopKAccuracyMetric
from utils.losses import get_criterion
from utils.optim import get_optimizer, get_scheduler, WarmUpLR
import models


# GPU settings
assert torch.cuda.is_available()
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
device = torch.device("cuda")
torch.backends.cudnn.benchmark = True

# General loss functions
cross_entropy_loss = nn.CrossEntropyLoss()

def train(current_epoch, model, train_loader, optimizer, criterion, scheduler, warmup_scheduler, args):
    logging.info("training epoch {}".format(current_epoch+1))

    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc="Epoch {}".format(current_epoch), ncols=0)
    pbar.set_description('Epoch {}/{}'.format(current_epoch+1, args.epochs + args.warmup_epochs))

    model.train()
    losses = AverageMeter()

    acc_metric = TopKAccuracyMetric(topk=(1, 5))

    y_pred = []
    y_true = []
    for idx, sample in pbar:
        optimizer.zero_grad()
        images = sample["image"].cuda(non_blocking=True)
        labels = sample["label"].cuda(non_blocking=True)

        y_pred = model(images)
        loss = criterion(y_pred, labels)
        losses.update(loss.item(), images.size(0))

        loss.backward()
        optimizer.step()

        if current_epoch < args.warmup_epochs:
            warmup_scheduler.step()

        if args.sched_mode == 'step' and current_epoch >= args.warmup_epochs: 
            scheduler.step()

        epoch_acc = acc_metric(y_pred, labels)

        # outputs = torch.argmax(outputs.softmax(-1), dim=1).tolist()

        # y_pred.extend(outputs)
        # y_true.extend(labels.tolist())

        lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix({'lr': lr, 'loss':losses.avg, 'acc': epoch_acc})
    
    pbar.close()
    # epoch_acc = acc_metric(torch.tensor(y_pred), torch.tensor(y_true))
    # a = y_pred == y_true
    # acc = torch.sum(a)
    # logging.info("lr: {}, train_loss: {}, acc: {}".format(lr, losses.avg, acc / len(y_true)))
    # logging.info("lr: {}, train_loss: {}".format(lr, losses.avg))
    return y_true, y_pred

def validate(model, val_loader, criterion, args):
    logging.info("Test phase")
    model.eval()

    probs = defaultdict(list)
    targets = defaultdict(list)
    losses = AverageMeter()
    acc_metric = TopKAccuracyMetric(topk=(1, 5))

    with torch.no_grad():
        for sample in tqdm(val_loader):
            images = sample["image"].cuda(non_blocking=True)
            labels = sample["label"].cuda(non_blocking=True)
            image_names = sample["image_name"]

            y_pred = model(images)

            loss = criterion(y_pred, labels)
            losses.update(loss.item(), images.size(0))

            epoch_acc = acc_metric(y_pred, labels)

            # for i in range(outputs.shape[0]):
            #     image_name = image_names[i]
            #     probs[image_name].append(outputs[i].tolist())
            #     targets[image_name].append(labels[i].tolist())
    
    logging.info("validate acc: {}".format(epoch_acc))
    return epoch_acc[0]

def parse_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    # log
    arg('--logdir',         type=str, default='logs')
    # Data
    arg('--data-dir',       type=str, default='CUB_200_2011')
    arg('--batch_size',     type=int, default=16)
    arg('--num_classes',    type=int, default=200)
    arg('--image_size',     type=int, default=384)
    arg('--tag',            type=str, default='bird')

    arg('--classifier',     type=str, default='Classifier')
    arg('--backbone',       type=str, default='vit_base_patch16_384')
    arg('--pretrained',     type=bool, default=True)
    arg('--dropout_rate',   type=float, default=0.5)
    arg('--resume',         type=str, default='')

    arg('--epochs',         type=int, default=30)
    arg('--warmup_epochs',  type=int, default=1)
    arg('--lr',             type=float, default=0.01)
    # Optim
    arg('--optim',          type=str, default='sgd', help='sgd, adam, adamw')
    arg('--sched',          type=str, default='cosine', help='step, cosine, plateau')
    # Loss
    arg('--focal_loss',     action='store_true')
    arg('--label_smoothing', type=float, default=0.1)
    arg('--workers',        type=int, default=8)

    # StepLR 参数
    arg('--sched_mode',     type=str, default='epoch', help='step, epoch')
    arg('--step_size',      type=int, default=1)
    arg('--gamma',          type=float, default=0.9)
    # ReduceLROnPlateau 参数
    arg('--patience',       type=int, default=1)
    arg('--factor',         type=float, default=0.1)

    # DevMode
    arg('--dev_mode',       type=bool, default=False)

    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    ###################################
    # Logs Setting
    ###################################
    localtime = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time()))
    log_dir = os.path.join(args.logdir, args.classifier + '_' + args.backbone, "{}".format(localtime))
    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(filename=os.path.join(log_dir, "train.log"), filemode='w', level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())

    for key, value in args.__dict__.items():
        logging.info('{} = {}'.format(key, value))
        with open(os.path.join(log_dir, 'config.json'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)

    ###################################
    # Datasets
    ###################################
    train_dataset, validate_dataset = get_trainval_dataset(args)
    train_loader    = DataLoader(train_dataset,     batch_size=args.batch_size,     shuffle=True,   num_workers=args.workers, pin_memory=True)
    validate_loader = DataLoader(validate_dataset,  batch_size=args.batch_size * 4, shuffle=False,  num_workers=args.workers, pin_memory=True)
    
    ###################################
    # Initialize Model
    ###################################
    model = models.__dict__[args.classifier](**args.__dict__).cuda()

    ###################################
    # Use Cuda
    ###################################
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model).cuda()

    ###################################
    # Resume CheckPoint
    ###################################
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            logging.info("=> loading checkpoint {}".format(args.resume))
            checkpoint = torch.load(args.resume)
            state_dict = checkpoint['state_dict']
            state_dict = {k:v for k, v in state_dict.items()}
            model.load_state_dict(state_dict, strict=False)
            logging.info("=> loaded checkpoint {} (epoch {}, best_score {})".format(args.resume, checkpoint['epoch'], checkpoint['best_acc']))

            start_epoch = checkpoint['epoch']
        else:
            logging.info("=> no checkpoint found at {}".format(args.finetune))

    ###################################
    # Optimizer, LR Scheduler, Loss
    ###################################
    optimizer = get_optimizer(args.optim, model, args)
    criterion = get_criterion(args)

    iter_per_epoch = len(train_loader)
    if args.sched_mode == 'step':
        args.epochs = iter_per_epoch * args.epochs

    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warmup_epochs)
    scheduler = get_scheduler(args.sched, optimizer, args)

    ###################################
    # Train, Validate
    ###################################
    best_acc = 0
    for epoch in range(start_epoch, args.epochs + args.warmup_epochs):
        # callback.on_epoch_begin()

        # pbar = tqdm(total=len(train_loader), unit='batches')
        #pbar.set_description('Epoch {}/{}'.format(epoch+1, args.epochs))

        train(epoch, model, train_loader, optimizer, criterion, scheduler, warmup_scheduler, args)
        epoch_acc = validate(model, validate_loader, criterion, args)

        if epoch >= args.warmup_epochs and args.sched_mode == 'epoch':
            if args.sched == 'plateau':
                # TODO
                # scheduler.step(epoch+1, metric=?)
                scheduler.step(epoch + 1)
            else:
                scheduler.step()
        
        best_acc = max(best_acc, epoch_acc)
        logging.info("best acc: {}".format(best_acc))
        # callback.on_epoch_end(logs, model)
        # pbar.close()


if __name__ == '__main__':
    main()
