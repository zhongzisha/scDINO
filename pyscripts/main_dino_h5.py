# main_dino_h5.py

import argparse
import os
import sys
import datetime
import time
import math
import json
import h5py
import io
from pathlib import Path
import argparse
import os
from random import shuffle
import sys
import datetime
import time
import math
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision import models as torchvision_models
from vision_transformer import DINOHead
import utils
import vision_transformer as vits
from tifffile import imread
import numpy
from PIL import Image
import torchvision
import ast
from catalyst.data import DistributedSamplerWrapper

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from torchvision import datasets, transforms
from torchvision import models as torchvision_models
from vision_transformer import DINOHead
import vision_transformer as vits
import tifffile

from catalyst.data import DistributedSamplerWrapper
import utils

# List of available torchvision architectures
torchvision_archs = sorted(
    name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name])
)

def get_args_parser():
    parser = argparse.ArgumentParser('DINO', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='vit_small', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base', 'xcit', 'deit_tiny', 'deit_small'] + torchvision_archs,
        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches).""")
    parser.add_argument('--out_dim', default=65536, type=int, help="""Dimensionality of
        the DINO head output.""")
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
        help="""Whether or not to weight normalize the last layer of the DINO head.""")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
        parameter for teacher update.""")
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
        help="Whether to use batch normalizations in projection head (Default: False)")

    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
        help="""Initial value for the teacher temperature.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
        of the teacher temperature.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=30, type=int,
        help='Number of warmup epochs for the teacher temperature (Default: 30).')

    # Training/Optimization parameters
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to restart from.")
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True, help="""Whether or not
        to use half precision for training.""")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping.""")
    parser.add_argument('--batch_size_per_gpu', default=64, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed.""")
    parser.add_argument("--lr", default=0.0003, type=float, help="""Learning rate at the end of
        linear warmup.""")
    parser.add_argument("--warmup_epochs", default=10, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer.""")

    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="Stochastic depth rate")
    parser.add_argument('--h5_paths', nargs='+', default=None, help='Paths to one or multiple h5 files with images')
    parser.add_argument('--h5_group', default='full', type=str, help='Group name in h5 files to access images')
    parser.add_argument('--dataset_dir', default=None, type=str, help='Path to the dataset directory of images')
    parser.add_argument('--selected_channels', default=[0,1,2], type=int, nargs='+', help="List of channel indices to use")
    parser.add_argument('--norm_per_channel_file', default=None, type=str, help="Path to mean and std file in json format")
    parser.add_argument('--upscale_factor', default=0, type=float, help='Upscale factor to upsample images')
    parser.add_argument('--center_crop', default=0, type=int, help='Center crop size to crop images')
    parser.add_argument('--images_are_RGB', default=False, type=utils.bool_flag, help='Whether images are RGB')
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.4, 1.0),
        help='Scale range for global crops')
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4),
        help='Scale range for local crops')
    parser.add_argument('--local_crops_number', type=int, default=8, help='Number of local crops')
    parser.add_argument('--train_datasetsplit_fraction', default=0.9, type=float,
        help='Fraction of data to use for training')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loader workers per GPU')
    parser.add_argument('--output_dir', default='./output', type=str, help='Path to save output')
    parser.add_argument('--name_of_run', default='run1', type=str, help='Name of the run')
    parser.add_argument('--full_ViT_name', default='vit_small', type=str, help='Full ViT name')
    parser.add_argument('--parse_params', default=None, type=str, help='Additional parameters to parse')
    parser.add_argument('--saveckp_freq', default=20, type=int, help='Save checkpoint every X epochs')

    # Distributed Training Arguments
    parser.add_argument("--dist_url", default="env://", type=str,
        help="""URL used to set up distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int,
        help="Please ignore and do not set this argument.")

    return parser

def train_dino(args, save_dir):
    utils.init_distributed_mode(args)  # Initialize distributed training
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True
    print('Saving log file of run parameters')
    with open(os.path.join(save_dir, "run_log.txt"), "w") as f:
        f.write(f"Successfully computed features with seed {args.seed}: \n")
        f.write("Parameters: \n")
        for arg in vars(args):
            f.write(f"{arg} : {getattr(args, arg)} \n")

    # ============ preparing data ... ============
    def load_mean_std_per_channel(norm_per_channel_file):
        with open(norm_per_channel_file) as f:
            norm_per_channel_json = json.load(f)
            norm_per_channel = [norm_per_channel_json['mean'], norm_per_channel_json['std']]
        return norm_per_channel

    def create_mean_std_per_channel_for_channel_combi(norm_per_channel_file, selected_channels):
        norm_per_channel = load_mean_std_per_channel(norm_per_channel_file)
    
        # Ensure that the number of normalization entries matches the number of selected channels
        if len(norm_per_channel[0]) != len(selected_channels) or len(norm_per_channel[1]) != len(selected_channels):
            raise ValueError(
                f"Number of normalization entries (mean: {len(norm_per_channel[0])}, std: {len(norm_per_channel[1])}) "
                f"does not match number of selected channels ({len(selected_channels)})."
            )
    
        # Assign normalization values in order, assuming they correspond to the selected channels
        mean_for_selected_channel = tuple(norm_per_channel[0])
        std_for_selected_channel = tuple(norm_per_channel[1])
    
        return mean_for_selected_channel, std_for_selected_channel
    
    if args.h5_paths is None and args.dataset_dir is None:
        raise ValueError("Please provide either --h5_paths or --dataset_dir")

    selected_channels = list(map(int, args.selected_channels))
    mean_for_selected_channel, std_for_selected_channel = create_mean_std_per_channel_for_channel_combi(args.norm_per_channel_file, selected_channels)

    transform = DataAugmentationDINO(
        args.images_are_RGB,
        args.global_crops_scale,
        args.local_crops_scale,
        args.local_crops_number,
        mean_for_selected_channel,
        std_for_selected_channel,
    )

    if args.h5_paths:
        # Using H5Dataset
        dataset_total = H5Dataset(
            args.h5_paths,
            group=args.h5_group,
            transform=transform,
            selected_channels=selected_channels,
            center_crop=args.center_crop
        )
    elif args.dataset_dir:
        # Using folder of images
        if not args.images_are_RGB:
            class Multichannel_dataset(datasets.ImageFolder):
                def __init__(self, root, transform=None, target_transform=None):
                    super().__init__(root, transform=transform, target_transform=target_transform)

                def __getitem__(self, idx):
                    path, target = self.samples[idx]
                    image_np = tifffile.imread(path)
                    image_np = image_np.astype(float)
                    image_np = image_np[selected_channels, :, :]  # Select channels without transposing
                    if args.center_crop:
                        image = torch.from_numpy(image_np)
                        transform_center_crop = transforms.CenterCrop(args.center_crop)
                        image = transform_center_crop(image)
                        image_np = image.detach().cpu().numpy()
                    image_np = utils.normalize_numpy_0_to_1_silent(image_np)
                    if utils.check_nan_silent(image_np):
                        print("NaN in image: ", path)
                        return None
                    else:
                        image = torch.from_numpy(image_np).float()  # No permute needed
                        if self.transform is not None:
                            image = self.transform(image)
                        if self.target_transform is not None:
                            target = self.target_transform(target)
                        return image, idx

            dataset_total = Multichannel_dataset(args.dataset_dir, transform=transform)
        else:
            dataset_total = datasets.ImageFolder(args.dataset_dir, transform=transform)

    # SAMPLER SECTION
    validation_split = float(1 - args.train_datasetsplit_fraction)
    shuffle_dataset = True
    dataset_size = len(dataset_total)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(args.seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    print("Split between train and test dataset:", len(train_indices), len(val_indices))
    print(f"Train dataset consists of {len(train_indices)} images.")

    # Creating data samplers and loaders:
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    train_sampler_wrapped = DistributedSamplerWrapper(train_sampler)  # Wrap with DistributedSampler

    data_loader = torch.utils.data.DataLoader(
        dataset_total,
        sampler=train_sampler_wrapped,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
        collate_fn=utils.collate_fn)

    print("Successfully loaded data.")

    # ============ building student and teacher networks ... ============
    print("Building student and teacher networks...")
    args.arch = args.arch.replace("deit", "vit")
    if args.arch in vits.__dict__.keys():
        student = vits.__dict__[args.arch](
            patch_size=args.patch_size,
            drop_path_rate=args.drop_path_rate,
            in_chans=len(selected_channels)
        )
        teacher = vits.__dict__[args.arch](patch_size=args.patch_size, in_chans=len(selected_channels))
        embed_dim = student.embed_dim
    elif args.arch in torchvision_models.__dict__.keys():
        student = torchvision_models.__dict__[args.arch]()
        teacher = torchvision_models.__dict__[args.arch]()
        embed_dim = student.fc.weight.shape[1]
    else:
        print(f"Unknown architecture: {args.arch}")
        sys.exit(1)

    # Multi-crop wrapper
    student = utils.MultiCropWrapper(student, DINOHead(
        embed_dim,
        args.out_dim,
        use_bn=args.use_bn_in_head,
        norm_last_layer=args.norm_last_layer,
    ))
    teacher = utils.MultiCropWrapper(
        teacher,
        DINOHead(embed_dim, args.out_dim, args.use_bn_in_head),
    )
    student, teacher = student.cuda(), teacher.cuda()

    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
        teacher_without_ddp = teacher.module
    else:
        teacher_without_ddp = teacher

    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])
    teacher_without_ddp.load_state_dict(student.module.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False

    print(f"Student and Teacher are built: they are both {args.arch} network.")
    print("Preparing loss, optimizer, and schedulers...")

    # ============ preparing loss ... ============
    dino_loss = DINOLoss(
        args.out_dim,
        args.local_crops_number + 2,  # total number of crops = 2 global + local
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
    ).cuda()

    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(student)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")

    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,
        args.min_lr,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader),
    )
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1,
                                               args.epochs, len(data_loader))
    print(f"Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ... ============
    start_epoch = 0
    start_time = time.time()
    print("Starting DINO training!")

    for epoch in range(start_epoch, args.epochs):
        data_loader.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(student, teacher, teacher_without_ddp, dino_loss,
                                      data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
                                      epoch, fp16_scaler, args)
        # ============ writing logs ... ============
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': vars(args),
            'dino_loss': dino_loss.state_dict(),
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(save_dir, 'checkpoint.pth'))
        if args.saveckp_freq and (epoch + 1) % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(save_dir, f'checkpoint{str(epoch+1).zfill(4)}.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch + 1}
        if utils.is_main_process():
            with (Path(save_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(student, teacher, teacher_without_ddp, dino_loss, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule, epoch,
                    fp16_scaler, args):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch + 1, args.epochs)
    for it, (images, _) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        it = len(data_loader) * epoch + it
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:
                param_group["weight_decay"] = wd_schedule[it]
        images = [im.cuda(non_blocking=True) for im in images if im is not None]
        if len(images) == 0:
            continue  # Skip if no valid images
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            teacher_output = teacher(images[:2])  # only the 2 global views pass through the teacher
            student_output = student(images)
            loss = dino_loss(student_output, teacher_output, epoch)

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        optimizer.zero_grad()
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)
                utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        with torch.no_grad():
            m = momentum_schedule[it]
            for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class DataAugmentationDINO(object):
    def __init__(self, images_are_RGB, global_crops_scale, local_crops_scale, local_crops_number, mean_for_selected_channel, std_for_selected_channel):
        if not images_are_RGB:
            flip_gamma_brightness = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                utils.AdjustBrightness(0.8),
            ])
            normalize = transforms.Compose([
                transforms.Normalize(mean_for_selected_channel, std_for_selected_channel),
            ])

            # first global crop
            self.global_transfo1 = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC),
                flip_gamma_brightness,
                utils.GaussianBlur_forGreyscaleMultiChan(1.0),
                normalize,
            ])
            # second global crop
            self.global_transfo2 = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC),
                flip_gamma_brightness,
                utils.GaussianBlur_forGreyscaleMultiChan(0.1),
                utils.Solarization_forGreyscaleMultiChan(0.2),
                normalize,
            ])
            # transformation for the local small crops
            self.local_crops_number = local_crops_number
            self.local_transfo = transforms.Compose([
                transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC),
                flip_gamma_brightness,
                utils.GaussianBlur_forGreyscaleMultiChan(0.5),
                normalize,
            ])
        else:
            flip_and_color_jitter = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
            ])
            normalize = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean_for_selected_channel, std_for_selected_channel),
            ])

            # first global crop
            self.global_transfo1 = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC),
                flip_and_color_jitter,
                utils.GaussianBlur(1.0),
                normalize,
            ])
            # second global crop
            self.global_transfo2 = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC),
                flip_and_color_jitter,
                utils.GaussianBlur(0.1),
                utils.Solarization(0.2),
                normalize,
            ])
            # transformation for the local small crops
            self.local_crops_number = local_crops_number
            self.local_transfo = transforms.Compose([
                transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC),
                flip_and_color_jitter,
                utils.GaussianBlur(p=0.5),
                normalize,
            ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        # Optionally, check for NaNs
        for crop in crops:
            if torch.isnan(crop).any():
                print("NaN found in the crop")
        return crops


class H5Dataset(torch.utils.data.Dataset):
    def __init__(self, h5_paths, group='full', transform=None, selected_channels=None, center_crop=None):
        self.h5_paths = h5_paths
        self.group = group
        self.transform = transform
        self.selected_channels = selected_channels
        self.center_crop = center_crop
        self.image_info = []
        for h5_index, h5_path in enumerate(self.h5_paths):
            with h5py.File(h5_path, 'r') as f:
                group_data = f[self.group]
                dataset_names = list(group_data.keys())
                for dataset_name in dataset_names:
                    self.image_info.append((h5_index, dataset_name))
        # To ensure that each worker opens its own file handle
        self.file_handles = {}

    def __len__(self):
        return len(self.image_info)

    def __getitem__(self, idx):
        h5_index, dataset_name = self.image_info[idx]
        h5_path = self.h5_paths[h5_index]
        if h5_path not in self.file_handles:
            self.file_handles[h5_path] = h5py.File(h5_path, 'r')
        group_data = self.file_handles[h5_path][self.group]
        tiff_bytes = group_data[dataset_name][()]
        with io.BytesIO(tiff_bytes) as bio:
            img_array = tifffile.imread(bio)
        image_np = img_array.astype(float)
        if self.selected_channels is not None:
            image_np = image_np[self.selected_channels, :, :]
        if self.center_crop:
            image = torch.from_numpy(image_np)
            transform_center_crop = transforms.CenterCrop(self.center_crop)
            image = transform_center_crop(image)
            image_np = image.detach().cpu().numpy()
        image_np = utils.normalize_numpy_0_to_1_silent(image_np)
        if utils.check_nan_silent(image_np):
            print("NaN in image: ", dataset_name)
            return None
        else:
            image = torch.from_numpy(image_np).float()
            if self.transform is not None:
                image = self.transform(image)
            return image, idx

    def __del__(self):
        for handle in self.file_handles.values():
            handle.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
    print("Parsing arguments")
    args = parser.parse_args()

    # Handle additional parameters
    if args.parse_params:
        try:
            # Use json.loads for safer and standard parsing
            additional_params = json.loads(args.parse_params)
            for key, value in additional_params.items():
                setattr(args, key, value)
        except json.JSONDecodeError as e:
            print(f"Error parsing --parse_params: {e}")
            sys.exit(1)

    # Define save directory
    save_dir = f"{args.output_dir}/{args.name_of_run}/scDINO_ViTs/{args.full_ViT_name}"
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    print("Starting train_dino!")
    train_dino(args, save_dir)
