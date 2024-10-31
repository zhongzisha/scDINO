import argparse
import os
import sys
import json
import h5py
import io
from pathlib import Path
from tqdm import tqdm

import numpy as np
from tifffile import imread

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import transforms

import vision_transformer as vits
import utils  

import re  
import pandas as pd 

def get_xy_coordinates(h5_file):
    """
    Extracts cell IDs and their coordinates from the H5 file's metadata.
    Assumes that 'cell_metadata' dataset exists with:
        - cell_id at index 1
        - x_coord at index 2
        - y_coord at index 3
    """
    with h5py.File(h5_file, 'r') as f:
        if 'cell_metadata' not in f:
            print(f"Error: 'cell_metadata' dataset not found in {h5_file}.")
            sys.exit(1)
        cell_metadata = f['cell_metadata'][:]
        # Extract cell IDs and coordinates
        cell_ids = cell_metadata[:, 1].astype(str)  # Ensure cell IDs are strings
        x_coords = cell_metadata[:, 2]
        y_coords = cell_metadata[:, 3]
    return cell_ids, x_coords, y_coords

class H5ImageDataset(torch.utils.data.Dataset):
    """
    Custom dataset to read images from H5 files.
    Assumes that images and cell_metadata are stored in the same H5 file and correspond by order.
    """

    def __init__(self, h5_paths, group='full', transform=None, selected_channels=None, center_crop=None, repeat_channels=1):
        self.h5_paths = h5_paths
        self.group = group
        self.transform = transform
        self.selected_channels = selected_channels
        self.center_crop = center_crop
        self.repeat_channels = repeat_channels  # New parameter to repeat channels
        self.image_info = []  # List of tuples: (h5_index, dataset_name)
        self.cell_ids = []    # Flattened list of cell_ids corresponding to image_info
        self.file_handles = {}  # Initialize here to prevent AttributeError

        for h5_index, h5_path in enumerate(self.h5_paths):
            try:
                with h5py.File(h5_path, 'r') as f:
                    if self.group not in f:
                        raise KeyError(f"Group '{self.group}' not found in {h5_path}.")
                    group_data = f[self.group]

                    # **Sort dataset_names numerically based on extracted number**
                    dataset_names = list(group_data.keys())
                    try:
                        dataset_names_sorted = sorted(
                            dataset_names,
                            key=lambda x: self.extract_num(x)
                        )
                    except ValueError as e:
                        raise ValueError(f"Error: {e}")

                    self.image_info.extend([(h5_index, ds_name) for ds_name in dataset_names_sorted])

                    # Extract cell_ids for this H5 file
                    cell_ids, _, _ = get_xy_coordinates(h5_path)
                    if len(cell_ids) != len(dataset_names_sorted):
                        raise ValueError(
                            f"Number of cell_ids ({len(cell_ids)}) does not match number of images ({len(dataset_names_sorted)}) in {h5_path}."
                        )
                    self.cell_ids.extend(cell_ids)  # Flatten the cell_ids
            except Exception as e:
                print(e)
                # Continue processing other H5 files instead of exiting
                continue

    def extract_num(self, dataset_name):
        """
        Extracts numerical index from dataset name.
        Assumes dataset names are in the format 'image{number}_all.tif'
        """
        match = re.search(r'image(\d+)_all\.tif', dataset_name)
        if match:
            return int(match.group(1))
        else:
            raise ValueError(f"Cannot extract numerical index from dataset name '{dataset_name}'.")

    def __len__(self):
        return len(self.image_info)

    def __getitem__(self, idx):
        h5_index, dataset_name = self.image_info[idx]
        h5_path = self.h5_paths[h5_index]
        if h5_path not in self.file_handles:
            try:
                self.file_handles[h5_path] = h5py.File(h5_path, 'r')
            except Exception as e:
                print(f"Error opening H5 file {h5_path}: {e}")
                return None
        group_data = self.file_handles[h5_path][self.group]
        tiff_bytes = group_data[dataset_name][()]
        with io.BytesIO(tiff_bytes) as bio:
            try:
                img_array = imread(bio)
            except Exception as e:
                print(f"Error reading image {dataset_name} in {h5_path}: {e}")
                return None

        image_np = img_array.astype(float)
        if self.selected_channels is not None:
            # Ensure single channel is selected and repeated
            # Assuming selected_channels is a list with one channel
            if len(self.selected_channels) != 1:
                print(f"Error: selected_channels should contain exactly one channel, got {self.selected_channels}")
                return None
            channel = self.selected_channels[0]
            if image_np.shape[0] < channel + 1:
                print(f"Warning: Image {dataset_name} has fewer channels than selected channel {channel}. Skipping.")
                return None
            # Select the specified channel
            image_np = image_np[channel, :, :]
            # Repeat the channel to create multiple channels (quintuple)
            image_np = np.stack([image_np] * self.repeat_channels, axis=0)  # e.g., 5 channels

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
            return image, idx  # Return idx for direct mapping

    def __del__(self):
        # Safely handle the deletion of file handles
        try:
            for handle in self.file_handles.values():
                handle.close()
        except AttributeError:
            # In case self.file_handles was never initialized
            pass

def custom_collate(batch):
    """
    Custom collate function to handle None entries and separate images and indices.
    """
    # Filter out None values
    batch = [item for item in batch if item is not None]
    if not batch:
        # Return empty tensors/lists if the batch is empty after filtering
        return torch.Tensor(), []
    
    # Unzip the batch into separate lists
    images, indices = zip(*batch)
    
    # Stack images into a single tensor
    images = torch.stack(images, dim=0)
    
    return images, indices

def save_features_to_csv(features, image_ids, output_csv_path):
    """
    Saves embeddings and image IDs to a CSV file.
    Each row corresponds to an image, and columns correspond to embedding dimensions.
    """
    # Convert features to numpy
    features_np = features.numpy()
    # Create a DataFrame with image_ids and features
    df = pd.DataFrame(features_np)
    df.insert(0, 'image_id', image_ids)
    # Save to CSV
    df.to_csv(output_csv_path, index=False)
    print(f"Features saved to {output_csv_path}")

@torch.no_grad()
def extract_features(model, data_loader, use_cuda=True):
    """
    Extracts features from the model for all images in the data_loader.
    """
    features = []
    indices = []
    total_batches = len(data_loader)

    with tqdm(total=total_batches, desc="Extracting features", unit="batch") as pbar:
        for samples, batch_indices in data_loader:
            if samples.nelement() == 0:
                # Skip empty batches
                pbar.update(1)
                continue
            if use_cuda:
                samples = samples.cuda(non_blocking=True)
            if samples.dim() == 3:
                samples = samples.unsqueeze(0)
            feats = model(samples.float()).clone()

            features.append(feats.cpu())
            indices.extend(batch_indices)

            pbar.update(1)

    if features:
        features = torch.cat(features, dim=0)
    else:
        features = torch.Tensor()
    return features, indices

def extract_and_save_feature_pipeline(args):
    """
    Main pipeline to extract features and save them with corresponding metadata.
    """
    torch.manual_seed(args.seed)

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
    ])

    if args.normalize:
        if args.norm_per_channel_file:
            try:
                with open(args.norm_per_channel_file) as f:
                    norm_per_channel_json = json.load(f)
                    norm_per_channel = [tuple(norm_per_channel_json['mean']), tuple(norm_per_channel_json['std'])]
                    print(f"Loaded normalization parameters from {args.norm_per_channel_file}")
            except Exception as e:
                print(f"Error reading --norm_per_channel_file: {e}")
                sys.exit(1)
        else:
            try:
                norm_per_channel = json.loads(args.norm_per_channel)
                print("Loaded normalization parameters from --norm_per_channel argument.")
            except json.JSONDecodeError:
                print("Warning: Could not parse --norm_per_channel. Using default values.")
                norm_per_channel = [(0,) * len(args.selected_channels), (1,) * len(args.selected_channels)]

        mean_for_selected_channel = [norm_per_channel[0][i] for i in range(len(args.selected_channels))]
        std_for_selected_channel = [norm_per_channel[1][i] for i in range(len(args.selected_channels))]
        print("Normalizing with mean:", mean_for_selected_channel, "and std:", std_for_selected_channel)
        transform.transforms.append(transforms.Normalize(mean=mean_for_selected_channel, std=std_for_selected_channel))
    else:
        print("Not applying normalization.")

    # Initialize dataset
    dataset_total = H5ImageDataset(
        h5_paths=args.h5_paths,
        group=args.h5_group,
        transform=transform,
        selected_channels=args.selected_channels,
        center_crop=args.center_crop,
        repeat_channels=5  # Quintupling the input channel
    )

    # Create data loader
    data_loader = torch.utils.data.DataLoader(
        dataset_total,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
        collate_fn=custom_collate  # Use the custom collate function
    )

    print("Data loader created")

    # Build model
    num_in_chans_pretrained = utils.get_pretrained_weights_in_chans(args.pretrained_weights)
    print(f"Pretrained weights have {num_in_chans_pretrained} input channels")
    model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0, in_chans=int(num_in_chans_pretrained))
    model.cuda()
    utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
    model.eval()
    print(f"Model {args.arch} {args.patch_size}x{args.patch_size} with {num_in_chans_pretrained} in_chans built.")

    # Adjust ViT model if needed
    if args.use_mean_patch_embedding:
        average_conv2d_weights = torch.mean(model.patch_embed.proj.weight, 1, keepdim=True)
        conv2d_weights_per_chan = average_conv2d_weights.repeat(1, len(args.selected_channels), 1, 1)
        model.patch_embed.proj.weight = nn.Parameter(conv2d_weights_per_chan)
        print("Adjusted patch embedding to use mean weights.")
    elif args.use_custom_embedding_map:
        embedding_seq = utils.embedding_seq(args.custom_embedding_map, args.selected_channels)
        model = utils.build_weight_emb(embedding_seq, model)
        print("Adjusted patch embedding to use custom embedding map.")

    # Extract cell IDs from each H5 file (already done in H5ImageDataset)
    cell_ids_list = dataset_total.cell_ids  # Flattened list of cell_ids
    print(f"Collected cell_ids from {len(args.h5_paths)} H5 files.")

    # Extract features
    print("Extracting features for dataset...")
    features, batch_indices = extract_features(model, data_loader, args.use_cuda)

    if features.nelement() == 0:
        print("No features extracted. Exiting.")
        sys.exit(1)

    features = nn.functional.normalize(features, dim=1, p=2)

    # Map indices to cell_ids and image_ids directly using the flattened lists
    cell_ids_mapped = [dataset_total.cell_ids[idx] for idx in batch_indices]
    image_ids = [dataset_total.image_info[idx][1] for idx in batch_indices]

    print(f"Total features extracted: {features.shape[0]}")
    print(f"Total cell_ids mapped: {len(cell_ids_mapped)}")
    print(f"Total image_ids mapped: {len(image_ids)}")

    # Save embeddings and mappings to CSV file
    # Construct output CSV path
    output_csv_path = os.path.join(args.output_dir, f"EA_017_scDINO_features_channel_{args.channel}.csv")
    save_features_to_csv(features, image_ids, output_csv_path)

def main():
    parser = argparse.ArgumentParser('Computation of CLS features')
    # Add necessary arguments
    parser.add_argument('--h5_paths', nargs='+', required=True, help='Paths to one or multiple H5 files with images and metadata (exclude normalized H5 files)')
    parser.add_argument('--h5_group', default='full', type=str, help='Group name in H5 files to access images')
    parser.add_argument('--name_of_run', default='recent_run', type=str)
    parser.add_argument('--batch_size_per_gpu', default=32, type=int, help='Per-GPU batch size')
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument('--use_cuda', default=True, type=utils.bool_flag,
                        help="Should we store the features on GPU? We recommend setting this to False if you encounter OOM")
    parser.add_argument('--arch', default='vit_small', type=str, help='Architecture')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
                        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--dataset_dir', default=None, type=str, help='Path to the dataset directory of images (not used if h5_paths are provided)')
    parser.add_argument('--images_are_RGB', help='If images are RGB, set this to True. If images are grayscale, set this to False.', default=False, type=utils.bool_flag)
    # Removed 'selected_channels' argument since we are processing one channel at a time
    # parser.add_argument('--selected_channels', nargs='+', type=int, default=[0, 1, 2, 3, 4], help="List of channel indexes of the .tiff images which should be used to create the tensors.")
    parser.add_argument('--channel', type=int, required=True, help='Channel index to process')  # New argument
    parser.add_argument('--channel_dict', type=str, help="Name of the channels in format as dict channel_number, channel_name.")
    parser.add_argument('--resize', default=False, help="If images should be resized")
    parser.add_argument('--resize_length', default=None, help="Quadratic resize length to resize images")
    parser.add_argument('--norm_per_channel', default="[(0, 0, 0, 0, 0), (1, 1, 1, 1, 1)]", type=str, help="2x tuple of mean and std per channel typically values between 0 and 1")
    parser.add_argument('--norm_per_channel_file', default=None, help="Path to file with mean and std per channel in JSON format.")
    parser.add_argument('--center_crop', default=None, type=int, help="Center crop size to crop images")
    parser.add_argument('--normalize', default=False, type=utils.bool_flag, help="Whether to normalize with mean and std per channel")
    parser.add_argument('--patch_embedding_mapping', default=None, help="Change the patch embedding weights by inputting a string of the sequence of rearrangement of the model '[0,1,2]' or the string 'average_weights' or None")
    parser.add_argument('--parse_params', help='Load settings from file in JSON format. Command line options override values in file.')
    parser.add_argument('--use_weighted_sampler', default=False, type=bool, help='Use weighted sampler for training.')
    parser.add_argument('--class_weights', default=None, help="List of weights for each class")
    parser.add_argument("--num_samples", default=None, type=int, help="Number of images to run in total.")
    parser.add_argument("--read_model_arch_dynamically", default=None, type=str, help="Read model architecture from pretrained weights")
    parser.add_argument("--use_mean_patch_embedding", default=False, type=bool, help="Use mean patch embedding instead of first patch embedding")
    parser.add_argument("--use_custom_embedding_map", default=False, type=bool, help="Use custom embedding map")
    parser.add_argument("--custom_embedding_map", default=None, type=str, help="Custom embedding map")
    parser.add_argument("--scDINO_full_pipeline", default=False, type=bool, help="Using scDINO full pipeline")
    parser.add_argument('--full_ViT_name', default='full_vit_name', type=str, help='Name channel combi ViT')
    parser.add_argument("--train_datasetsplit_fraction", default=0.8, type=float, help="When using scDINO full pipeline")
    parser.add_argument("--test_datasetsplit_fraction", default=0.8, type=float, help="When using downstream analysis only")
    parser.add_argument('--seed', default=42, type=int, help='Random seed.')
    parser.add_argument('--folder_depth_for_labels', default=0, type=int, help='Folder depth for labels. 0 means that the labels are the folder names where the images are stored. 1 means one level above and so on. e.g., path/to/images/labelwhen3/labelwhen2/labelwhen1/labelwhen0/image.tiff')
    parser.add_argument('--output_dir', default='.', type=str)
    # Changed output_h5 to output_csv, handled inside the pipeline
    # parser.add_argument('--output_h5', default='embeddings.h5', type=str, help='Output HDF5 file to store embeddings and cell IDs')
    args = parser.parse_args()

    # Handle additional parameters from JSON
    if args.parse_params:
        try:
            with open(args.parse_params, 'r') as f:
                additional_params = json.load(f)
            for key, value in additional_params.items():
                setattr(args, key, value)
            print(f"Loaded additional parameters from {args.parse_params}")
        except json.JSONDecodeError as e:
            print(f"Error parsing --parse_params: {e}")
            sys.exit(1)
        except FileNotFoundError:
            print(f"Error: The file specified in --parse_params does not exist.")
            sys.exit(1)

    # Read mean and std per channel from file if provided
    if args.norm_per_channel_file:
        try:
            with open(args.norm_per_channel_file) as f:
                norm_per_channel_json = json.load(f)
                norm_per_channel = [tuple(norm_per_channel_json['mean']), tuple(norm_per_channel_json['std'])]
                args.norm_per_channel = norm_per_channel
                print(f"Loaded normalization parameters from {args.norm_per_channel_file}")
        except Exception as e:
            print(f"Error reading --norm_per_channel_file: {e}")
            sys.exit(1)

    # Adjust model architecture based on pretrained weights if necessary
    def adjust_model_architecture(args):
        model_name = os.path.basename(args.pretrained_weights)
        number = re.findall(r'\d+', model_name)
        if number:
            args.patch_size = int(number[0])
        if "small" in model_name.lower():
            args.arch = "vit_small"
        elif "base" in model_name.lower():
            args.arch = "vit_base"
        print(f"Adjusted model architecture to {args.arch} with patch size {args.patch_size} based on pretrained weights.")

    if args.read_model_arch_dynamically:
        adjust_model_architecture(args)

    # Set up GPU
    if args.use_cuda and torch.cuda.is_available():
        args.gpu = 0
        torch.cuda.set_device(args.gpu)
        cudnn.benchmark = True
        print("CUDA is enabled. Using GPU.")
    else:
        args.gpu = -1  # Use CPU
        print("CUDA is disabled or not available. Using CPU.")

    # Print all arguments for verification
    print("\n----- Script Arguments -----")
    for k, v in sorted(dict(vars(args)).items()):
        print(f"{k}: {v}")
    print("-----------------------------")

    # Set selected_channels to the specified channel, repeated 1 time (we will handle repetition in dataset)
    args.selected_channels = [args.channel]

    # Extract features and get cell IDs
    extract_and_save_feature_pipeline(args)

    # Save embeddings and mappings to CSV file is handled within the pipeline

if __name__ == '__main__':
    main()
