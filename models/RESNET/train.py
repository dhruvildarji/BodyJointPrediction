import os
import torch
import torchvision.transforms as transforms
from dataset.ego4d_dataset import ego4dDataset
from models.resnet_pose_3d import load_pretrained_weights, ResNetDirectRegressionPose
from tensorboardX import SummaryWriter
from tqdm import tqdm
from utils.functions import (
    AverageMeter,
    create_logger,
    parse_args_function,
    update_config,
)
from utils.loss import Pose3DLoss, L1Pose3DLoss, L1Pose3DLossWithThreshold


def denormalize(tensor, mean, std):
    """
    Denormalize a tensor image with mean and standard deviation.
    Args:
        tensor (Tensor): The normalized tensor image.
        mean (list): The mean used for normalization.
        std (list): The standard deviation used for normalization.
    Returns:
        Tensor: The denormalized tensor image.
    """
    mean = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device)
    std = torch.tensor(std, dtype=tensor.dtype, device=tensor.device)
    tensor = tensor * std[None, :, None, None] + mean[None, :, None, None]
    return tensor


def get_feature_maps(model, x):
    feature_maps = {}
    x = model.resnet.conv1(x)
    x = model.resnet.bn1(x)
    x = model.resnet.relu(x)
    x = model.resnet.maxpool(x)

    feature_maps['layer1'] = model.resnet.layer1(x)
    feature_maps['layer2'] = model.resnet.layer2(feature_maps['layer1'])
    feature_maps['layer3'] = model.resnet.layer3(feature_maps['layer2'])
    feature_maps['layer4'] = model.resnet.layer4(feature_maps['layer3'])

    return feature_maps


def visualize_feature_maps(writer, layer_name, feature_maps, global_steps):
    # Take the first 3 channels to visualize if the number of channels is greater than 3
    num_channels = min(feature_maps.shape[1], 3)
    feature_maps = feature_maps[:, :num_channels, :, :]
    writer.add_images(f'Train/{layer_name}_FeatureMaps',
                      feature_maps[:4].cpu(), global_steps)


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

    # switch to train mode
    model.train()
    train_loader = tqdm(train_loader, dynamic_ncols=True)
    print_interval = len(train_loader) // config.TRAIN_PRINT_NUM

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

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

            if writer_dict:
                writer = writer_dict["writer"]
                global_steps = writer_dict["train_global_steps"]
                writer.add_scalar("Loss/train", loss_3d.avg, global_steps)
                writer_dict["train_global_steps"] = global_steps + 1

                # Denormalize images before logging
                input_denorm = denormalize(input[:4].cpu(), mean, std)

                # Log denormalized images
                writer.add_images('Train/Images', input_denorm, global_steps)

                # Log predictions and ground truth
                for j in range(min(4, len(pose_3d_pred))):
                    writer.add_text(
                        f'Train/Predicted Pose {j}', str(pose_3d_pred[j].detach().cpu().numpy()), global_steps)
                    writer.add_text(
                        f'Train/Ground Truth Pose {j}', str(pose_3d_gt[j].cpu().numpy()), global_steps)

                # Log weights and gradients
                for name, param in model.named_parameters():
                    writer.add_histogram(
                        f'Weights/{name}', param, global_steps)
                    if param.grad is not None:
                        writer.add_histogram(
                            f'Gradients/{name}', param.grad, global_steps)

                # Log feature maps from relevant layers
                feature_maps = get_feature_maps(model, input)
                for layer_name, layer_output in feature_maps.items():
                    visualize_feature_maps(
                        writer, layer_name, layer_output, global_steps)


def validate(
    val_loader, model, criterion, device, logger, writer_dict
):
    loss_3d = AverageMeter()

    # switch to evaluate mode
    model.eval()

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

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

            if writer_dict:
                writer = writer_dict["writer"]
                global_steps = writer_dict["valid_global_steps"]
                if i == 0:
                    input_denorm = denormalize(input[:4].cpu(), mean, std)
                    writer.add_images('Val/Images', input_denorm, global_steps)
                    for j in range(min(4, len(pose_3d_pred))):
                        writer.add_text(
                            f'Val/Predicted Pose {j}', str(pose_3d_pred[j].detach().cpu().numpy()), global_steps)
                        writer.add_text(
                            f'Val/Ground Truth Pose {j}', str(pose_3d_gt[j].cpu().numpy()), global_steps)

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

def get_loss_function(config):
    loss_name = config.TRAIN.LOSS.lower()
    
    if loss_name == 'pose3dloss':
        return Pose3DLoss()
    elif loss_name == 'l1pose3dloss':
        return L1Pose3DLoss()
    elif loss_name == 'l1pose3dlosswiththresh':
        return L1Pose3DLossWithThreshold(config.TRAIN.ERROR_THRESHOLD)
    else:
        raise ValueError(f"Unsupported loss function: {config.TRAIN.LOSS}")


def get_optimizer(config, model_parameters):
    if config.TRAIN.OPTIMIZER.lower() == 'adam':
        return torch.optim.Adam(model_parameters, lr=config.TRAIN.LR, weight_decay=config.TRAIN.WEIGHT_DECAY)
    elif config.TRAIN.OPTIMIZER.lower() == 'nag':
        return torch.optim.SGD(model_parameters, lr=config.TRAIN.LR, momentum=config.TRAIN.MOMENTUM, weight_decay=config.TRAIN.WEIGHT_DECAY, nesterov=True)
    else:
        raise ValueError(f"Unsupported optimizer type: {config.TRAIN.OPTIMIZER}")

def main(args):
    torch.cuda.empty_cache()
    cfg = update_config(args.cfg_file)
    device = torch.device(
        f"cuda:{args.gpu_number[0]}" if torch.cuda.is_available() else "cpu"
    )
    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg_file, "train")

    ############ MODEL ###########
    model = ResNetDirectRegressionPose(**cfg.MODEL)
    # Load pretrained weights if available
    if args.pretrained_ckpt:
        load_pretrained_weights(
            model, torch.load(args.pretrained_ckpt, map_location=device)
        )
        logger.info(f"Loaded pretrained weight from {args.pretrained_ckpt}")

    model = model.to(device)
    writer_dict = {
        "writer": SummaryWriter(log_dir=tb_log_dir),
        "train_global_steps": 0,
        "valid_global_steps": 0,
    }

    # Log the model graph
    dummy_input = torch.zeros((1, 3, 224, 224)).to(device)
    writer_dict["writer"].add_graph(model, dummy_input)

    ############# CRITERION AND OPTIMIZER ###########
    # define loss function (criterion) and optimizer
    # criterion = Pose3DLoss().cuda()
    criterion = get_loss_function(cfg).cuda()
    # optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    optimizer = get_optimizer(cfg, model.parameters())

    ########### DATASET ###########
    # Load Ego4D dataset
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225]),
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
        shuffle=False,
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
                    final_output_dir, f"RESNET-HandPose-{cfg.DATASET.DATASET}.pt"
                ),
            )


if __name__ == "__main__":
    args = parse_args_function()
    main(args)
