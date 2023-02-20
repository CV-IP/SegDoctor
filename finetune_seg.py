import os
import torch
import datetime
import numpy as np
from tqdm import tqdm
import albumentations as A
from dataset import SegDataset
from metric import Evaluator
from model.unet import UNet
from model.unet_resnet import UNet_ResNet
from torch.nn import CrossEntropyLoss
import torch.distributed as dist
from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2
from util import set_seed, reduce_tensor, setup_for_distributed, link_loss3
import warnings
import argparse
warnings.filterwarnings("ignore")

from train_utils import *
from model.spixel_loss import *

def get_train_transforms(args):
    return A.Compose([
            #A.RandomCrop(height=512, width=1024),    # cityscapes
            A.SmallestMaxSize(max_size=512, interpolation=1, always_apply=False, p=1),    # voc
            A.RandomCrop(height=512, width=512),                                          # voc
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Normalize(max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.)

def get_val_transforms(args):
    return A.Compose([
            A.SmallestMaxSize(max_size=512, interpolation=1, always_apply=False, p=1),   # voc
            A.Normalize(max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.)

# 计算验证集IoU
@torch.no_grad()
def val_model(model, loader, device):
    evaluator = Evaluator(args.num_classes)

    model.eval()                     #冻结模型中的Bn和Dropout
    tbar = loader
    for iter, (inputs, labels, _) in enumerate(tbar):
        inputs, labels = inputs.to(device), labels.to(device)
        out, _, _ = model(inputs)
        out = torch.argmax(out, dim=1)
        evaluator.add_batch(labels, out)

        if iter % args.print_freq == 0:
            print(f'[Val] {iter} / {len(loader)}')
    
    confusion_matrix = evaluator.export_tensor().to(device)
    dist.all_reduce(confusion_matrix, op=dist.reduce_op.SUM)

    evaluator.set_confusion_matrix(confusion_matrix)
    mIoU = evaluator.Mean_Intersection_over_Union()
    meanIoU = np.mean(mIoU)
    print(f'Class IoU: {mIoU}')
    print(f'Mean IoU: {meanIoU}')
    print(f'FWMIoU: ', evaluator.Frequency_Weighted_Intersection_over_Union())
    print(f'PA: ', evaluator.Pixel_Accuracy())
    print(f'MPA: ', evaluator.Pixel_Accuracy_Class())
    return meanIoU

def train_model(model, criterion, optimizer, lr_scheduler=None, train_loader=None, val_loader=None, device=None, start_epoch=1):
    best_score = 0
    best_epoch = 0

    # 开始训练
    for epoch in range(start_epoch, args.epochs + 1):
        loss_epoch = 0
        model.train()

        if args.use_ddp:
            train_loader.sampler.set_epoch(epoch)

        # tbar = tqdm(train_loader, desc=f'Epoch [{epoch} / {args.epochs}]', dynamic_ncols=True)
        tbar = train_loader

        for iter, (inputs, labels, _) in enumerate(tbar):
            inputs, labels = inputs.to(device), labels.to(device)
            
            inputs = inputs.float()

            out, xs, prob = model(inputs)       # [N, C, h, W]
            pred = torch.argmax(out, dim=1)
            
            loss = 0
            subloss = 0
            for feat, ps in zip([xs[0]], [16]):   # pspnet 8 deeplab 8 unet 16
                subloss = subloss + link_loss3(labels.squeeze().long(), pred, feat, ps=ps)
            loss = loss + subloss + criterion(out.float(), labels.long())
            
            ########
            label_1hot = label2one_hot_torch(labels, C=22) # set C=50 as SSN does
            spixelID, xy_feat = init_spixel_grid(args, downsize=4)
            LABXY_feat_tensor = build_LABXY_feat(label_1hot, xy_feat)  # B* (50+2 )* H * W
            slic_loss, loss_sem, loss_pos = compute_semantic_pos_loss(prob, LABXY_feat_tensor,
                                                                pos_weight=0.003, kernel_size=8)
            loss += 0.01 * slic_loss
            # print(slic_loss)
            # print(loss)
            ########
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if args.use_ddp:
                loss_epoch += reduce_tensor(loss)
            else:
                loss_epoch += loss.item()
        
            lr_scheduler.step()

            if iter % args.print_freq == 0 or iter == len(tbar):
                print(f'[{epoch} / {args.epochs}] [{iter} / {len(train_loader)}]' + \
                      f'loss={loss_epoch / (1 + iter)} ' + \
                    #   f'subloss={subloss} ' + \
                      f'lr={lr_scheduler.get_last_lr()[0]} '
                )

        mIoU = val_model(model, val_loader, device)
        print(f'Epoch: {epoch}, mIoU: {mIoU}')

        if args.LOCAL_RANK in (-1, 0):
            if args.use_ddp:
                modelName = model.module.__class__.__name__
            else:
                modelName = model.__class__.__name__

            best_model_path = args.model_save_dir + f'{modelName}_best_finetune_{args.img_size[0]}' + '.pth'
            if best_score < mIoU:
                best_score = mIoU
                best_epoch = epoch
                state = {
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': lr_scheduler.state_dict(),
                }
                torch.save(state, best_model_path)
                print("Best fold/epoch/score: {} / {} to {}".format(best_epoch, best_score, best_model_path))

def main(args):

    if args.use_ddp:
        torch.cuda.set_device(args.LOCAL_RANK)
    else:
        torch.cuda.set_device(args.gpu_id)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    

    if args.use_ddp:
        torch.distributed.init_process_group('nccl', world_size=args.WORLD_SIZE, rank=args.LOCAL_RANK, timeout=datetime.timedelta(seconds=3600))  # 1小时

    if args.LOCAL_RANK in (-1, 0):
        os.makedirs(args.model_save_dir, exist_ok=True, mode=0o777)
        os.makedirs(args.log_save_dir, exist_ok=True, mode=0o777)
        os.makedirs(args.fig_save_dir, exist_ok=True, mode=0o777)

    set_seed(args.seed)
    setup_for_distributed()
    print('[Device]', device)

    train_dataset = SegDataset(transforms=get_train_transforms(args), mode='train')
    val_dataset = SegDataset(transforms=get_val_transforms(args), mode='val')

    if args.use_ddp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, shuffle=False, num_workers=4, drop_last=True)
        
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
        val_loader = DataLoader(val_dataset, batch_size=1, sampler=val_sampler, shuffle=False, num_workers=4)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

    # 初始化模型
    if args.use_ddp:
        model = UNet_ResNet(num_classes=args.num_classes).to(device)     

        if args.sync_bn:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    else:
        model = UNet_ResNet(num_classes=args.num_classes).to(device)

    # 优化器,学习率策略，损失函数
    if args.use_ddp:
        optimizer = torch.optim.SGD(model.module.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-3)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-3)
        
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader), eta_min=0)
    
    start_epoch = 1
    if args.resume:
        state_dict = torch.load(args.resume)
        start_epoch = state_dict['epoch'] + 1
        print('Resume Model Load State Dict', model.load_state_dict(state_dict['model']))      
        print('Resume Optimizer Load State Dict', optimizer.load_state_dict(state_dict['optimizer']))
        print('Resume Scheduler Load State Dict', scheduler.load_state_dict(state_dict['scheduler']))
    
    criterion = CrossEntropyLoss(ignore_index=255)
    # 训练模型
    try:
        train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, device, start_epoch)
    except KeyboardInterrupt:
        print('Saved interrupt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # runtime
    parser.add_argument('--gpu_id', type=int, default=3, help='device id')

    # hyps
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--use_ddp', action='store_true', default=False)
    parser.add_argument('--sync_bn', action='store_true', default=False)
    parser.add_argument('--LOCAL_RANK', type=int, default=os.getenv('LOCAL_RANK', -1))
    parser.add_argument('--WORLD_SIZE', type=int, default=os.getenv('WORLD_SIZE', 1))

    # experiment setting
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--num_classes', type=int, default=21)
    parser.add_argument('--print_freq', type=int, default=100)
    parser.add_argument('--img_size', type=list, default=[512, 512])
    parser.add_argument('--load_weight', type=str, default='runs/weights/UNet_ResNet_best_512.pth')

    # result save-setting
    parser.add_argument('--log_save_dir', type=str, default='./runs/logs/')
    parser.add_argument('--fig_save_dir', type=str, default='./runs/figs/')
    parser.add_argument('--model_save_dir', type=str, default='./runs/weights/')

    args = parser.parse_args()
    main(args)
