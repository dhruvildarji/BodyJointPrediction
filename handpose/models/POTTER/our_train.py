import os

import torch
import torchvision.transforms as transforms
from dataset.ego4d_dataset import ego4dDataset
# from models.PoolAttnHR_Pose_3D import load_pretrained_weights, PoolAttnHR_Pose_3D
from models.sample_model import SimpleCNN, SimpleCNNResNet
from tensorboardX import SummaryWriter
from tqdm import tqdm
from utils.functions import (
    AverageMeter,
    create_logger,
    parse_args_function,
    update_config,
)
from utils.loss import Pose3DLoss
from utils.SampleLoss import SimplePose3DLoss
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import torchvision
import cv2
torch.autograd.set_detect_anomaly(True)


def save_checkpoint(state, is_best, checkpoint_dir, filename='checkpoint.pth.tar'):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    torch.save(state, checkpoint_path)
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best_model.pth.tar')
        torch.save(state, best_path)


# Function to add images to TensorBoard
def add_images(writer, features, global_step):
    for name, feature in features.items():
        if len(feature.shape) == 4:  # Check if the tensor is 4D (batch_size, channels, height, width)
            # Take the first 8 feature maps
            feature = feature[:8]
            # Normalize the feature maps for better visualization
            feature = (feature - feature.min()) / (feature.max() - feature.min())
            # Ensure the feature maps have 3 channels
            if feature.shape[1] == 1:
                feature = feature.repeat(1, 3, 1, 1)  # Repeat the channel dimension if it has only 1 channel
            elif feature.shape[1] == 2:
                # If it has 2 channels, add an empty third channel
                feature = torch.cat([feature, torch.zeros_like(feature[:, :1, :, :])], dim=1)
            elif feature.shape[1] > 3:
                feature = feature[:, :3, :, :]  # Select the first 3 channels if more than 3 channels

            # Resize the feature maps to a consistent size for visualization
            feature = F.interpolate(feature, size=(128, 128), mode='bilinear', align_corners=False)
            feature = (feature * 255).byte()  # Convert to uint8 for visualization
            grid = torchvision.utils.make_grid(feature, nrow=4, normalize=False)
            writer.add_image(f'{name}_features', grid, global_step)


def train(
    config,
    train_loader,
    model,
    criterion,
    optimizer,
    epoch,
    device,
    logger,
    writer_dict,
):
    loss_3d = AverageMeter()
    writer = writer_dict['writer']
    global_steps = writer_dict["train_global_steps"]

    # switch to train mode
    model.train()
    train_loader = tqdm(train_loader, dynamic_ncols=True)
    print_interval = len(train_loader) // config.TRAIN_PRINT_NUM

    for i, (input, pose_3d_gt, vis_flag, _) in enumerate(train_loader):
        # compute output
        input = input.to(device)
        pose_3d_pred = model(input)

        # Assign None kpts as zero
        pose_3d_gt[~vis_flag] = 0
        pose_3d_gt = pose_3d_gt.to(device)
        vis_flag = vis_flag.to(device)

        pose_3d_loss = criterion(pose_3d_pred, pose_3d_gt, vis_flag)
        loss = pose_3d_loss



        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        loss_3d.update(pose_3d_loss.item())

        # Log info
        if (i + 1) % print_interval == 0:
            msg = (
                "Epoch: [{0}][{1}/{2}]\t"
                "3D Loss {loss_3d.val:.5f} ({loss_3d.avg:.5f})".format(
                    epoch, i + 1, len(train_loader), loss_3d=loss_3d
                )
            )
            logger.info(msg)

            # global_steps = writer_dict["train_global_steps"]
            # writer.add_scalar("Loss/train", loss_3d.avg, global_steps)

            # add_images(writer, features, global_steps)

            features = model.get_features()

            add_images(writer, features, global_steps)

            if writer_dict:
                writer = writer_dict["writer"]
                global_steps = writer_dict["train_global_steps"]
                writer.add_scalar("Loss/train", loss_3d.avg, global_steps)
                writer_dict["train_global_steps"] = global_steps + 1
    writer_dict['writer'].close()

def validate(
    val_loader, model, criterion, device, logger, writer_dict
):
    loss_3d = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        val_loader = tqdm(val_loader, dynamic_ncols=True)
        for i, (input, pose_3d_gt, vis_flag, _) in enumerate(val_loader):
            # compute output
            input = input.to(device)
            pose_3d_pred = model(input)
            pose_3d_gt[~vis_flag] = 0
            pose_3d_gt = pose_3d_gt.to(device)
            vis_flag = vis_flag.to(device)

            pose_3d_loss = criterion(pose_3d_pred, pose_3d_gt, vis_flag)

            # measure accuracy and record loss
            loss_3d.update(pose_3d_loss.item())

        # Log info
        msg = (
            "Val: [{0}/{1}]\t"
            "3D Loss {loss_3d.avg:.5f}".format(
                i + 1, len(val_loader), loss_3d=loss_3d
            )
        )
        logger.info(msg)

        if writer_dict:
            writer = writer_dict["writer"]
            global_steps = writer_dict["valid_global_steps"]
            writer.add_scalar("Loss/val", loss_3d.avg, global_steps)
            writer_dict["valid_global_steps"] = global_steps + 1

    return loss_3d.avg



def main(args):
    torch.cuda.empty_cache()
    cfg = update_config(args.cfg_file)
    pretrained_hand_pose_CKPT = False
    device = torch.device(
        f"cuda:{args.gpu_number[0]}" if torch.cuda.is_available() else "cpu"
    )
    logger, final_output_dir, tb_log_dir = create_logger(cfg, args.cfg_file, "train")

    ############ MODEL ###########
    model = SimpleCNN()
    # model = SimpleCNNResNet()

    model = model.to(device)

    # input_tensor = torch.randn(1, 3, 256, 256)  # Batch size 1, 3 channels (RGB), 256x256 image
    # output = model(input_tensor)
    # print(output.shape)  # Expected output shape: (1, 21, 64, 3)

    writer_dict = {
        "writer": SummaryWriter(log_dir=tb_log_dir),
        "train_global_steps": 0,
        "valid_global_steps": 0,
    }

    ########### DATASET ###########
    # Load Ego4D dataset
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = ego4dDataset(args, cfg, split="train", transform=transform)
    valid_dataset = ego4dDataset(args, cfg, split="val", transform=transform)
    # Dataloader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=cfg.TRAIN.SHUFFLE,
        num_workers=cfg.WORKERS,
        pin_memory=True,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.WORKERS,
        pin_memory=True,
    )
    logger.info(f"Loaded ground truth annotation from {args.gt_anno_dir}")
    logger.info(
        f"Number of annotation(s): Train: {len(train_dataset)}\t Val: {len(valid_dataset)}"
    )
    logger.info(
        f"Learning rate: {cfg.TRAIN.LR} || Batch size: Train:{cfg.TRAIN.BATCH_SIZE}\t Val: {cfg.TEST.BATCH_SIZE}"
    )

    ############# CRITERION AND OPTIMIZER ###########
    # define loss function (criterion) and optimizer
    criterion = SimplePose3DLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.LR)


    ############ Train model & validation ###########
    best_val_loss = 1e2
    for epoch in range(cfg.TRAIN.BEGIN_EPOCH, cfg.TRAIN.END_EPOCH):
        # train for one epoch
        logger.info(f"############# Starting Epoch {epoch} #############")

        train(
            cfg,
            train_loader,
            model,
            criterion,
            optimizer,
            epoch,
            device,
            logger,
            writer_dict,
        )
        # evaluate on validation set
        val_loss = validate(
            valid_loader,
            model,
            criterion,
            device,
            logger,
            writer_dict,
        )

    
        # Save best model weight
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save model weight
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                },
                os.path.join(
                    final_output_dir, f"Sample-{cfg.DATASET.DATASET}.pt"
                ),
            )

        
        if epoch % 2 == 0:
            # Save model weight at the end of even epoch
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
            
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                },
                is_best,
                final_output_dir
            )


if __name__ == "__main__":
    args = parse_args_function()
    main(args)