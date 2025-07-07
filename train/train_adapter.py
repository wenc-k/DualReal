# Copyright 2024 The CogView team, Tsinghua University & ZhipuAI and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import sys
import json
from contextlib import contextmanager
import numpy as np
from copy import copy, deepcopy
import argparse
import logging
import math
import os
import os.path as osp
import shutil
from pathlib import Path
from typing import List, Optional, Tuple, Union
from PIL import Image
import colorama

import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from huggingface_hub import create_repo
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, T5EncoderModel, T5Tokenizer

from model.dit import CogVideoXTransformer3DModelWithAdapter
from model.pipeline import CogVideoXPipeline

import diffusers    
from diffusers import AutoencoderKLCogVideoX, CogVideoXDPMScheduler
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from diffusers.optimization import get_scheduler
from diffusers.pipelines.cogvideo.pipeline_cogvideox import get_resize_crop_region_for_grid
from diffusers.training_utils import (
    cast_training_params
)
from diffusers.utils import check_min_version, convert_unet_state_dict_to_peft, export_to_video, is_wandb_available
from diffusers.utils.torch_utils import is_compiled_module

from src.clip import FrozenOpenCLIPCustomEmbedder
import src.transforms as data 
import random

os.environ["WANDB_API_KEY"] = 'KEY'
os.environ["WANDB_MODE"] = "offline"

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.31.0.dev0")

logger = get_logger(__name__)

colorama.init(autoreset=True)
highlight_color = colorama.Fore.CYAN

class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, message):
        for f in self.files:
            f.write(message)

    def flush(self):
        for f in self.files:
            f.flush()

def get_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script for CogVideoX.")

    # Model information
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )

    # Dataset information
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) containing the training data of instance images (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--instance_data_root_id",
        type=str,
        default=None,
        help=("A folder containing the training data."),
    )
    parser.add_argument(
        "--instance_data_root_motion",
        type=str,
        default=None,
        help=("A folder containing the training data."),
    )
    parser.add_argument(
        "--video_column_id",
        type=str,
        default="video",
    )
    parser.add_argument(
        "--video_column_motion",
        type=str,
        default="video",
    )
    parser.add_argument(
        "--caption_column_id",
        type=str,
        default="text",
    )
    parser.add_argument(
        "--caption_column_motion",
        type=str,
        default="text",
    )
    parser.add_argument(
        "--id_token", type=str, default=None, help="Identifier token appended to the start of each prompt if provided."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )

    # Training information
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default='bf16',
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="cogvideox-lora",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="All input videos are resized to this height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=720,
        help="All input videos are resized to this width.",
    )
    parser.add_argument("--fps", type=int, default=8, help="All input videos will be used at this FPS.")
    parser.add_argument(
        "--max_num_frames", type=int, default=49, help="All input videos will be truncated to these many frames."
    )
    parser.add_argument(
        "--skip_frames_start",
        type=int,
        default=0,
        help="Number of frames to skip from the beginning of each input video. Useful if training data contains intro sequences.",
    )
    parser.add_argument(
        "--skip_frames_end",
        type=int,
        default=0,
        help="Number of frames to skip from the end of each input video. Useful if training data contains outro sequences.",
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip videos horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--train_batch_size_other", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides `--num_train_epochs`.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine_with_restarts",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=200, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--enable_slicing",
        action="store_true",
        default=False,
        help="Whether or not to use VAE slicing for saving memory.",
    )
    parser.add_argument(
        "--enable_tiling",
        action="store_true",
        default=False,
        help="Whether or not to use VAE tiling for saving memory.",
    )

    # Optimizer
    parser.add_argument(
        "--optimizer",
        type=lambda s: s.lower(),
        default="adamw",
        choices=["adam", "adamw", "prodigy"],
        help=("The optimizer type to use."),
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW",
    )
    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.95, help="The beta2 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--prodigy_beta3",
        type=float,
        default=None,
        help="Coefficients for computing the Prodigy optimizer's stepsize using running averages. If set to None, uses the value of square root of beta2.",
    )
    parser.add_argument("--prodigy_decouple", action="store_true", help="Use AdamW style decoupled weight decay")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-04, help="Weight decay to use for unet params")
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer and Prodigy optimizers.",
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--prodigy_use_bias_correction", action="store_true", help="Turn on Adam's bias correction.")
    parser.add_argument(
        "--prodigy_safeguard_warmup",
        action="store_true",
        help="Remove lr from the denominator of D estimate to avoid issues during warm-up stage.",
    )

    # Other information
    parser.add_argument("--tracker_name", type=str, default=None, help="Project tracker name")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help="Directory where logs are stored.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )

    # dualreal
    parser.add_argument(
        "--use_controller",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--hypernet_outputdim",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--close_grad_mask",
        type=bool,
        default=False,
    ) 
    parser.add_argument(
        "--motion_ratio",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--spatial_adapter_alpha",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--temporal_adapter_alpha",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--training_parameters",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--use_IdAdapter",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--use_MotionAdapter",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--use_id_condition",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--use_motion_condition",
        type=bool,
        default=False,
    )

    parser.add_argument(
        "--inference_steps",
        type=int,
        default=100000,
    )

    parser.add_argument(
        "--inf_eva_prompts",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--spatial_adapter_hidden_dim",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--temporal_adapter_hidden_dim",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--spatial_adapter_condition_dim",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--temporal_adapter_condition_dim",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--ref_image_path",
        type=str,
        default="",
    )

    parser.add_argument(
        "--clip_pretrained",
        type=str,
    )
    
    parser.add_argument(
        "--p_image_zero",
        type=float,
        default="0",
    )
    return parser.parse_args()

class VideoDataset_for_id(Dataset):
    def __init__(
        self,
        instance_data_root: Optional[str] = None,
        dataset_name: Optional[str] = None,
        dataset_config_name: Optional[str] = None,
        caption_column: str = "text",
        video_column: str = "video",
        height: int = 480,
        width: int = 720,
        fps: int = 8,
        max_num_frames: int = 49,
        skip_frames_start: int = 0,
        skip_frames_end: int = 0,
        cache_dir: Optional[str] = None,
        id_token: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.instance_data_root = Path(instance_data_root) if instance_data_root is not None else None
        self.dataset_name = dataset_name
        self.dataset_config_name = dataset_config_name
        self.caption_column = caption_column
        self.video_column = video_column
        self.height = height
        self.width = width
        self.fps = fps
        self.max_num_frames = max_num_frames
        self.skip_frames_start = skip_frames_start
        self.skip_frames_end = skip_frames_end
        self.cache_dir = cache_dir
        self.id_token = id_token or ""

        if dataset_name is not None:
            self.instance_prompts, self.instance_video_paths = self._load_dataset_from_hub()
        else:
            self.instance_prompts, self.instance_video_paths = self._load_dataset_from_local_path()

        self.num_instance_videos = len(self.instance_video_paths)
        if self.num_instance_videos != len(self.instance_prompts):
            raise ValueError(
                f"Expected length of instance prompts and videos to be the same but found {len(self.instance_prompts)=} and {len(self.instance_video_paths)=}. Please ensure that the number of caption prompts and videos match in your dataset."
            )
        videos, vit_frames = self._preprocess_data()
        self.instance_videos = videos
        self.instance_vit_frames = vit_frames # [num_instance_videos, C, H, W]

    def __len__(self):
        return self.num_instance_videos

    def __getitem__(self, index):
        return {
            "instance_prompt": self.id_token + self.instance_prompts[index],
            "instance_video": self.instance_videos[index],
            "instance_vit_frames": self.instance_vit_frames[index],
        }

    def _load_dataset_from_hub(self):
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "You are trying to load your data using the datasets library. If you wish to train using custom "
                "captions please install the datasets library: `pip install datasets`. If you wish to load a "
                "local folder containing images only, specify --instance_data_root instead."
            )

        # Downloading and loading a dataset from the hub. See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.0.0/en/dataset_script
        dataset = load_dataset(
            self.dataset_name,
            self.dataset_config_name,
            cache_dir=self.cache_dir,
        )
        column_names = dataset["train"].column_names

        if self.video_column is None:
            video_column = column_names[0]
            logger.info(f"`video_column` defaulting to {video_column}")
        else:
            video_column = self.video_column
            if video_column not in column_names:
                raise ValueError(
                    f"`--video_column` value '{video_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
                )

        if self.caption_column is None:
            caption_column = column_names[1]
            logger.info(f"`caption_column` defaulting to {caption_column}")
        else:
            caption_column = self.caption_column
            if self.caption_column not in column_names:
                raise ValueError(
                    f"`--caption_column` value '{self.caption_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
                )

        instance_prompts = dataset["train"][caption_column]
        instance_videos = [Path(self.instance_data_root, filepath) for filepath in dataset["train"][video_column]]

        return instance_prompts, instance_videos

    def _load_dataset_from_local_path(self):
        if not self.instance_data_root.exists():
            raise ValueError("Instance videos root folder does not exist")

        prompt_path = self.instance_data_root.joinpath(self.caption_column)
        video_path = self.instance_data_root.joinpath(self.video_column)

        if not prompt_path.exists() or not prompt_path.is_file():
            raise ValueError(
                "Expected `--caption_column` to be path to a file in `--instance_data_root` containing line-separated text prompts."
            )
        if not video_path.exists() or not video_path.is_file():
            raise ValueError(
                "Expected `--video_column` to be path to a file in `--instance_data_root` containing line-separated paths to video data in the same directory."
            )

        with open(prompt_path, "r", encoding="utf-8") as file:
            instance_prompts = [line.strip() for line in file.readlines() if len(line.strip()) > 0]
        with open(video_path, "r", encoding="utf-8") as file:
            instance_videos = [
                self.instance_data_root.joinpath(line.strip()) for line in file.readlines() if len(line.strip()) > 0
            ]

        if any(not path.is_file() for path in instance_videos):
            raise ValueError(
                "Expected '--video_column' to be a path to a file in `--instance_data_root` containing line-separated paths to video data but found atleast one path that is not a valid file."
            )

        return instance_prompts, instance_videos

    def _preprocess_data(self):
        try:
            import decord
        except ImportError:
            raise ImportError(
                "The `decord` package is required for loading the video dataset. Install with `pip install decord`"
            )

        decord.bridge.set_bridge("torch")

        videos = []
        vit_frames = []
        train_transforms = transforms.Compose(
            [
                transforms.Lambda(lambda x: x / 255.0 * 2.0 - 1.0),
            ]
        )

        for filename in self.instance_video_paths:
            video_reader = decord.VideoReader(uri=filename.as_posix(), width=self.width, height=self.height)
            video_reader_refimg = decord.VideoReader(uri=filename.as_posix(), width=224, height=224)
            video_num_frames = len(video_reader) # 1

            start_frame = min(self.skip_frames_start, video_num_frames)
            end_frame = max(0, video_num_frames - self.skip_frames_end)
            if end_frame <= start_frame:
                frames = video_reader.get_batch([start_frame])
                ref_frame = video_reader_refimg.get_batch([start_frame])
            elif end_frame - start_frame <= self.max_num_frames:
                frames = video_reader.get_batch(list(range(start_frame, end_frame)))
                ref_frame = video_reader_refimg.get_batch(list(range(start_frame, end_frame)))
            else:
                indices = list(range(start_frame, self.max_num_frames))
                frames = video_reader.get_batch(indices)
                ref_frame = video_reader_refimg.get_batch(indices)

            print("id dataset, length of ref_frame******", len(ref_frame))
            ref_idx = 0 # 只有一帧
            mid_frame = copy(ref_frame[ref_idx])
            mid_frame = mid_frame.float()
            vit_frame = train_transforms(mid_frame)
            
            # Ensure that we don't go over the limit
            frames = frames[: self.max_num_frames]
            selected_num_frames = frames.shape[0]

            # Choose first (4k + 1) frames as this is how many is required by the VAE
            remainder = (3 + (selected_num_frames % 4)) % 4
            if remainder != 0:
                frames = frames[:-remainder]
            selected_num_frames = frames.shape[0]

            assert (selected_num_frames - 1) % 4 == 0
            
            # Training transforms
            frames = frames.float()
            frames = torch.stack([train_transforms(frame) for frame in frames], dim=0)
            videos.append(frames.permute(0, 3, 1, 2).contiguous())  # [F, C, H, W]
            vit_frames.append(vit_frame.permute(2, 0, 1).contiguous()) # [C, H, W]

        return videos, vit_frames

class VideoDataset_for_motion(Dataset):
    def __init__(
        self,
        instance_data_root: Optional[str] = None,
        dataset_name: Optional[str] = None,
        dataset_config_name: Optional[str] = None,
        caption_column: str = "text",
        video_column: str = "video",
        height: int = 480,
        width: int = 720,
        fps: int = 8,
        max_num_frames: int = 49,
        skip_frames_start: int = 0,
        skip_frames_end: int = 0,
        cache_dir: Optional[str] = None,
        id_token: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.instance_data_root = Path(instance_data_root) if instance_data_root is not None else None
        self.dataset_name = dataset_name
        self.dataset_config_name = dataset_config_name
        self.caption_column = caption_column
        self.video_column = video_column
        self.height = height
        self.width = width
        self.fps = fps
        self.max_num_frames = max_num_frames
        self.skip_frames_start = skip_frames_start
        self.skip_frames_end = skip_frames_end
        self.cache_dir = cache_dir
        self.id_token = id_token or ""

        if dataset_name is not None:
            self.instance_prompts, self.instance_video_paths = self._load_dataset_from_hub()
        else:
            self.instance_prompts, self.instance_video_paths = self._load_dataset_from_local_path()

        self.num_instance_videos = len(self.instance_video_paths)
        if self.num_instance_videos != len(self.instance_prompts):
            raise ValueError(
                f"Expected length of instance prompts and videos to be the same but found {len(self.instance_prompts)=} and {len(self.instance_video_paths)=}. Please ensure that the number of caption prompts and videos match in your dataset."
            )
        videos, vit_frames = self._preprocess_data()
        self.instance_videos = videos
        self.instance_vit_frames = vit_frames

    def __len__(self):
        return self.num_instance_videos

    def __getitem__(self, index):
        return {
            "instance_prompt": self.id_token + self.instance_prompts[index],
            "instance_video": self.instance_videos[index],
            "instance_vit_frames": self.instance_vit_frames[index],
        }

    def _load_dataset_from_hub(self):
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "You are trying to load your data using the datasets library. If you wish to train using custom "
                "captions please install the datasets library: `pip install datasets`. If you wish to load a "
                "local folder containing images only, specify --instance_data_root instead."
            )

        # Downloading and loading a dataset from the hub. See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.0.0/en/dataset_script
        dataset = load_dataset(
            self.dataset_name,
            self.dataset_config_name,
            cache_dir=self.cache_dir,
        )
        column_names = dataset["train"].column_names

        if self.video_column is None:
            video_column = column_names[0]
            logger.info(f"`video_column` defaulting to {video_column}")
        else:
            video_column = self.video_column
            if video_column not in column_names:
                raise ValueError(
                    f"`--video_column` value '{video_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
                )

        if self.caption_column is None:
            caption_column = column_names[1]
            logger.info(f"`caption_column` defaulting to {caption_column}")
        else:
            caption_column = self.caption_column
            if self.caption_column not in column_names:
                raise ValueError(
                    f"`--caption_column` value '{self.caption_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
                )

        instance_prompts = dataset["train"][caption_column]
        instance_videos = [Path(self.instance_data_root, filepath) for filepath in dataset["train"][video_column]]

        return instance_prompts, instance_videos

    def _load_dataset_from_local_path(self):
        if not self.instance_data_root.exists():
            raise ValueError("Instance videos root folder does not exist")

        prompt_path = self.instance_data_root.joinpath(self.caption_column)
        video_path = self.instance_data_root.joinpath(self.video_column)

        if not prompt_path.exists() or not prompt_path.is_file():
            raise ValueError(
                "Expected `--caption_column` to be path to a file in `--instance_data_root` containing line-separated text prompts."
            )
        if not video_path.exists() or not video_path.is_file():
            raise ValueError(
                "Expected `--video_column` to be path to a file in `--instance_data_root` containing line-separated paths to video data in the same directory."
            )

        with open(prompt_path, "r", encoding="utf-8") as file:
            instance_prompts = [line.strip() for line in file.readlines() if len(line.strip()) > 0]
        with open(video_path, "r", encoding="utf-8") as file:
            instance_videos = [
                self.instance_data_root.joinpath(line.strip()) for line in file.readlines() if len(line.strip()) > 0
            ]

        if any(not path.is_file() for path in instance_videos):
            raise ValueError(
                "Expected '--video_column' to be a path to a file in `--instance_data_root` containing line-separated paths to video data but found atleast one path that is not a valid file."
            )

        return instance_prompts, instance_videos

    def _preprocess_data(self):
        try:
            import decord
        except ImportError:
            raise ImportError(
                "The `decord` package is required for loading the video dataset. Install with `pip install decord`"
            )

        decord.bridge.set_bridge("torch")

        videos = []
        vit_frames = []
        train_transforms = transforms.Compose(
            [
                transforms.Lambda(lambda x: x / 255.0 * 2.0 - 1.0),
            ]
        )

        for filename in self.instance_video_paths:
            video_reader = decord.VideoReader(uri=filename.as_posix(), width=self.width, height=self.height)
            video_reader_refimg = decord.VideoReader(uri=filename.as_posix(), width=224, height=224)
            video_num_frames = len(video_reader)

            start_frame = min(self.skip_frames_start, video_num_frames)
            end_frame = max(0, video_num_frames - self.skip_frames_end)
            if end_frame <= start_frame:
                frames = video_reader.get_batch([start_frame])
                ref_frame = video_reader_refimg.get_batch([start_frame])
            elif end_frame - start_frame <= self.max_num_frames:
                frames = video_reader.get_batch(list(range(start_frame, end_frame)))
                ref_frame = video_reader_refimg.get_batch(list(range(start_frame, end_frame)))
            else:
                indices = list(range(start_frame, self.max_num_frames))
                frames = video_reader.get_batch(indices)
                ref_frame = video_reader_refimg.get_batch(indices)

            # ref image for motion adapter
            ref_idx = np.random.randint(0, len(ref_frame))
            mid_frame = copy(ref_frame[ref_idx])
            mid_frame = mid_frame.float()
            vit_frame = train_transforms(mid_frame)

            # Ensure that we don't go over the limit
            frames = frames[: self.max_num_frames]
            selected_num_frames = frames.shape[0]

            # Choose first (4k + 1) frames as this is how many is required by the VAE
            remainder = (3 + (selected_num_frames % 4)) % 4
            if remainder != 0:
                frames = frames[:-remainder]
            selected_num_frames = frames.shape[0]

            assert (selected_num_frames - 1) % 4 == 0
            
            # Training transforms
            frames = frames.float()
            frames = torch.stack([train_transforms(frame) for frame in frames], dim=0)
            videos.append(frames.permute(0, 3, 1, 2).contiguous())  # [F, C, H, W]
            vit_frames.append(vit_frame.permute(2, 0, 1).contiguous()) # [C, F, H]

        return videos, vit_frames

def _get_t5_prompt_embeds(
    tokenizer: T5Tokenizer,
    text_encoder: T5EncoderModel,
    prompt: Union[str, List[str]],
    num_videos_per_prompt: int = 1,
    max_sequence_length: int = 226,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    text_input_ids=None,):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("`text_input_ids` must be provided when the tokenizer is not specified.")

    prompt_embeds = text_encoder(text_input_ids.to(device))[0]
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

    return prompt_embeds

def encode_prompt(
    tokenizer: T5Tokenizer,
    text_encoder: T5EncoderModel,
    prompt: Union[str, List[str]],
    num_videos_per_prompt: int = 1,
    max_sequence_length: int = 226,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    text_input_ids=None,):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    prompt_embeds = _get_t5_prompt_embeds(
        tokenizer,
        text_encoder,
        prompt=prompt,
        num_videos_per_prompt=num_videos_per_prompt,
        max_sequence_length=max_sequence_length,
        device=device,
        dtype=dtype,
        text_input_ids=text_input_ids,
    )
    return prompt_embeds

def compute_prompt_embeddings(tokenizer, text_encoder, prompt, max_sequence_length, device, dtype, requires_grad: bool = False):
    if requires_grad:
        prompt_embeds = encode_prompt(
            tokenizer,
            text_encoder,
            prompt,
            num_videos_per_prompt=1,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=dtype,
        )
    else:
        with torch.no_grad():
            prompt_embeds = encode_prompt(
                tokenizer,
                text_encoder,
                prompt,
                num_videos_per_prompt=1,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )
    return prompt_embeds

def prepare_rotary_positional_embeddings(
    height: int,
    width: int,
    num_frames: int,
    vae_scale_factor_spatial: int = 8,
    patch_size: int = 2,
    attention_head_dim: int = 64,
    device: Optional[torch.device] = None,
    base_height: int = 480,
    base_width: int = 720,
) -> Tuple[torch.Tensor, torch.Tensor]:
    grid_height = height // (vae_scale_factor_spatial * patch_size)
    grid_width = width // (vae_scale_factor_spatial * patch_size)
    base_size_width = base_width // (vae_scale_factor_spatial * patch_size)
    base_size_height = base_height // (vae_scale_factor_spatial * patch_size)

    grid_crops_coords = get_resize_crop_region_for_grid((grid_height, grid_width), base_size_width, base_size_height)
    freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
        embed_dim=attention_head_dim,
        crops_coords=grid_crops_coords,
        grid_size=(grid_height, grid_width),
        temporal_size=num_frames,
    )

    freqs_cos = freqs_cos.to(device=device)
    freqs_sin = freqs_sin.to(device=device)
    return freqs_cos, freqs_sin

def get_optimizer(args, params_to_optimize, use_deepspeed: bool = False):
    # Use DeepSpeed optimzer
    if use_deepspeed:
        from accelerate.utils import DummyOptim

        return DummyOptim(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
            weight_decay=args.adam_weight_decay,
        )

    # Optimizer creation
    supported_optimizers = ["adam", "adamw", "prodigy"]
    if args.optimizer not in supported_optimizers:
        logger.warning(
            f"Unsupported choice of optimizer: {args.optimizer}. Supported optimizers include {supported_optimizers}. Defaulting to AdamW"
        )
        args.optimizer = "adamw"

    if args.use_8bit_adam and not (args.optimizer.lower() not in ["adam", "adamw"]):
        logger.warning(
            f"use_8bit_adam is ignored when optimizer is not set to 'Adam' or 'AdamW'. Optimizer was "
            f"set to {args.optimizer.lower()}"
        )

    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

    if args.optimizer.lower() == "adamw":
        optimizer_class = bnb.optim.AdamW8bit if args.use_8bit_adam else torch.optim.AdamW

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
            weight_decay=args.adam_weight_decay,
        )
    elif args.optimizer.lower() == "adam":
        optimizer_class = bnb.optim.Adam8bit if args.use_8bit_adam else torch.optim.Adam

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
            weight_decay=args.adam_weight_decay,
        )
    elif args.optimizer.lower() == "prodigy":
        try:
            import prodigyopt
        except ImportError:
            raise ImportError("To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`")

        optimizer_class = prodigyopt.Prodigy

        if args.learning_rate <= 0.1:
            logger.warning(
                "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
            )

        optimizer = optimizer_class(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            beta3=args.prodigy_beta3,
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
            decouple=args.prodigy_decouple,
            use_bias_correction=args.prodigy_use_bias_correction,
            safeguard_warmup=args.prodigy_safeguard_warmup,
        )

    return optimizer

def log_variable_to_file(variable, path="log.txt"):
    """
    Logs a given variable to a specified file.
    
    Parameters:
    - variable: The variable to log (any data type).
    - path (str): The path to the log file. Defaults to 'log.txt'.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    with open(path, 'a', encoding='utf-8') as log_file:
        log_file.write(str(variable) + '\n'+ '\n')
    print(f"Variable logged to {path}")
    
def main(args):
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    log_file_path = Path(args.output_dir, "log_file.txt")

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
            ).repo_id

    load_dtype = torch.bfloat16 if "5b" in args.pretrained_model_name_or_path.lower() else torch.float16

    # Prepare models and scheduler
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )

    text_encoder = T5EncoderModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )

    vae = AutoencoderKLCogVideoX.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
    )

    scheduler = CogVideoXDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    transformer = CogVideoXTransformer3DModelWithAdapter.from_pretrained(  
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        torch_dtype=load_dtype,
        revision=args.revision,
        variant=args.variant,
        use_IdAdapter=args.use_IdAdapter,
        use_MotionAdapter=args.use_MotionAdapter,
        spatial_adapter_alpha=args.spatial_adapter_alpha,
        temporal_adapter_alpha=args.temporal_adapter_alpha,   
        spatial_adapter_hidden_dim=args.spatial_adapter_hidden_dim,
        temporal_adapter_hidden_dim=args.temporal_adapter_hidden_dim,
    )
    if args.use_IdAdapter:
        if args.use_id_condition:
            transformer.set_id_adapter(conditioning_dim=args.spatial_adapter_condition_dim)
            print("\033[1;31m set id adapter with id conditioning... \033[0m")
        else:
            transformer.set_id_adapter()
            print(highlight_color+"set id adapter...")

    if args.use_MotionAdapter:
        if args.use_motion_condition:
            transformer.set_motion_adapter(conditioning_dim=args.temporal_adapter_condition_dim)
            print("\033[1;31m set motion adapter with motion conditioning... \033[0m")
        else:
            transformer.set_motion_adapter()
            print(highlight_color+"set motion adapter...")

    if args.use_controller:
        transformer.set_controller(output_dim=args.hypernet_outputdim)
        print(highlight_color+f"set controller...") 

    print("transformer init successfully...")

    if args.enable_slicing:
        vae.enable_slicing()
    if args.enable_tiling:
        vae.enable_tiling()

    # We only train the additional adapter layers 
    text_encoder.requires_grad_(False)
    transformer.requires_grad_(False)
    vae.requires_grad_(False)

    if args.training_parameters:
        param_keys = args.training_parameters.split(',')
    else:
        param_keys = ['id_attn_adapter', 'id_ff_adapter', 'motion_attn_adapter', 'motion_ff_adapter', 'controller']
    
    print("training parameters:", param_keys)
    for name, param in transformer.named_parameters():
        if any(key in name for key in param_keys):
            param.requires_grad = True

    trainable_params = [name for name, param in transformer.named_parameters() if param.requires_grad]
    log_variable_to_file("Trainable parameters: {}".format(trainable_params), log_file_path)
    
    # For mixed precision training we cast all non-trainable weights (vae, text_encoder and transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.state.deepspeed_plugin:
        # DeepSpeed is handling precision, use what's in the DeepSpeed config
        if (
            "fp16" in accelerator.state.deepspeed_plugin.deepspeed_config
            and accelerator.state.deepspeed_plugin.deepspeed_config["fp16"]["enabled"]
        ):
            weight_dtype = torch.float16
        if (
            "bf16" in accelerator.state.deepspeed_plugin.deepspeed_config
            and accelerator.state.deepspeed_plugin.deepspeed_config["bf16"]["enabled"]
        ):
            weight_dtype = torch.float16
    else:
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    text_encoder.to(accelerator.device, dtype=weight_dtype)
    transformer.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
    
    # clip init
    image_encoder = FrozenOpenCLIPCustomEmbedder(args.clip_pretrained, device=accelerator.device)
    image_encoder.to(device=accelerator.device, dtype=weight_dtype)

    # zero_feature init.
    white_feature = image_encoder(image_encoder.white_image.to(weight_dtype))
    white_feature = white_feature.unsqueeze(1)
    zero_feature = torch.zeros_like(white_feature, device=white_feature.device) 
    
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            transformer_lora_layers_to_save = None

            for model in models:
                if isinstance(model, type(unwrap_model(transformer))):
                    transformer_lora_layers_to_save = get_peft_model_state_dict(model)
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

            CogVideoXPipeline.save_lora_weights(
                output_dir,
                transformer_lora_layers=transformer_lora_layers_to_save,
            )

    def load_model_hook(models, input_dir):
        transformer_ = None

        while len(models) > 0:
            model = models.pop()

            if isinstance(model, type(unwrap_model(transformer))):
                transformer_ = model
            else:
                raise ValueError(f"Unexpected save model: {model.__class__}")

        lora_state_dict = CogVideoXPipeline.lora_state_dict(input_dir)

        transformer_state_dict = {
            f'{k.replace("transformer.", "")}': v for k, v in lora_state_dict.items() if k.startswith("transformer.")
        }
        transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
        incompatible_keys = set_peft_model_state_dict(transformer_, transformer_state_dict, adapter_name="default")
        if incompatible_keys is not None:
            # check only for unexpected keys
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                logger.warning(
                    f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                    f" {unexpected_keys}. "
                )

        # Make sure the trainable params are in float32. This is again needed since the base models
        # are in `weight_dtype`. More details:
        # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
        if args.mixed_precision == "fp16":
            # only upcast trainable parameters (LoRA) into fp32
            cast_training_params([transformer_])

    def motion_ref_image_encoder(pretrained, image_path, device, dtype):
        # clip init
        image_encoder = FrozenOpenCLIPCustomEmbedder(pretrained, device=device)
        image_encoder.to(device=device, dtype=dtype)
        # load img
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        # trans img
        vit_trans = data.Compose([
            data.CenterCropWide(size=(256, 256)), # change if different img size
            data.Resize([224, 224]),
            data.ToTensor(),
            data.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])])

        with torch.no_grad():
            image_tensor = vit_trans(image)
            image_tensor = image_tensor.unsqueeze(0)
            y_visual = image_encoder(image=image_tensor.to(device=device, dtype=dtype))
            y_visual = y_visual.unsqueeze(1) # [1,1,1024]

        return y_visual.to(device=device, dtype=dtype)

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16":
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params([transformer], dtype=torch.float32)

    transformer_adapter_parameters = [p for p in transformer.parameters() if p.requires_grad]
    # Optimization parameters
    transformer_parameters_with_lr = {"params": transformer_adapter_parameters, "lr": args.learning_rate}
    params_to_optimize = [transformer_parameters_with_lr]

    use_deepspeed_optimizer = (
        accelerator.state.deepspeed_plugin is not None
        and "optimizer" in accelerator.state.deepspeed_plugin.deepspeed_config
    )
    use_deepspeed_scheduler = (
        accelerator.state.deepspeed_plugin is not None
        and "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config
    )

    optimizer = get_optimizer(args, params_to_optimize, use_deepspeed=use_deepspeed_optimizer)  

    # init dataset
    train_dataset_id = VideoDataset_for_id(
        instance_data_root=args.instance_data_root_id,
        dataset_name=args.dataset_name,
        dataset_config_name=args.dataset_config_name,
        caption_column=args.caption_column_id,
        video_column=args.video_column_id,
        height=args.height,
        width=args.width,
        fps=args.fps,
        max_num_frames=args.max_num_frames,
        skip_frames_start=args.skip_frames_start,
        skip_frames_end=args.skip_frames_end,
        cache_dir=args.cache_dir,
        id_token=args.id_token,
    )
    train_dataset_motion = VideoDataset_for_motion(
        instance_data_root=args.instance_data_root_motion,
        dataset_name=args.dataset_name,
        dataset_config_name=args.dataset_config_name,
        caption_column=args.caption_column_motion,
        video_column=args.video_column_motion,
        height=args.height,
        width=args.width,
        fps=args.fps,
        max_num_frames=args.max_num_frames,
        skip_frames_start=args.skip_frames_start,
        skip_frames_end=args.skip_frames_end,
        cache_dir=args.cache_dir,
        id_token=args.id_token,
    )

    def encode_video(video):
        video = video.to(accelerator.device, dtype=vae.dtype).unsqueeze(0)
        video = video.permute(0, 2, 1, 3, 4)  # [1, C, F, H, W]
        latent_dist = vae.encode(video).latent_dist
        return latent_dist
    
    def encode_video_frame_condition(frame):
        frame = frame.to(accelerator.device, dtype=weight_dtype).unsqueeze(0) #[1, 3, 224, 224]
        y_image = image_encoder(frame) # [1, 1024]
        y_image = y_image.unsqueeze(1).detach() # [1, 1, 1024]
        y_image[torch.rand(y_image.size(0)) < args.p_image_zero, :] = zero_feature # [1, 1, 1024]
        return y_image
    
    def derangement_shuffle(items):
        n = len(items)
        if n < 2:
            return items 
        indices = list(range(n))
        for i in range(n-1, 0, -1):
            j = random.randint(0, i-1)  
            indices[i], indices[j] = indices[j], indices[i]
        assert all(indices[i] != i for i in range(n)), "No swap, no misalignment"

        return [items[i] for i in indices]

    train_dataset_id.instance_videos = [encode_video(video) for video in train_dataset_id.instance_videos]
    train_dataset_id.instance_vit_frames = [encode_video_frame_condition(frame) for frame in train_dataset_id.instance_vit_frames]
    if args.use_id_condition:
        train_dataset_id.instance_vit_frames = derangement_shuffle(train_dataset_id.instance_vit_frames)

    train_dataset_motion.instance_videos = [encode_video(video) for video in train_dataset_motion.instance_videos]
    train_dataset_motion.instance_vit_frames = [encode_video_frame_condition(frame) for frame in train_dataset_motion.instance_vit_frames]

    def collate_fn(examples):
        videos = [example["instance_video"].sample() * vae.config.scaling_factor for example in examples] 
        prompts = [example["instance_prompt"] for example in examples]
        vit_frames = [example["instance_vit_frames"] for example in examples]

        videos = torch.cat(videos) # [batch_size, 16, F, 60, 90]
        vit_frames = torch.cat(vit_frames)
        videos = videos.to(memory_format=torch.contiguous_format).float()
        vit_frames = vit_frames.to(memory_format=torch.contiguous_format).float()

        return {
            "videos": videos,
            "prompts": prompts,
            "vit_frames": vit_frames,
        }
    
    def load_prompts(file_path: str) -> list[str]:
        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]

    def load_cogvideox_pipeline(args, dtype, transformer):
        cogvideox_tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
        )

        cogvideox_text_encoder = T5EncoderModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
        )
        cogvideox_text_encoder = cogvideox_text_encoder.to(dtype)

        cogvideox_vae = AutoencoderKLCogVideoX.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
        )
        cogvideox_vae = cogvideox_vae.to(dtype)

        cogvideox_scheduler = CogVideoXDPMScheduler.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="scheduler"
        )

        if transformer:
            print("use training transformer...")
            cogvideox_transformer=transformer

        pipe = CogVideoXPipeline(
            tokenizer=cogvideox_tokenizer,
            text_encoder=cogvideox_text_encoder,
            vae=cogvideox_vae,
            scheduler=cogvideox_scheduler,
            transformer=cogvideox_transformer,
        ) 
        return pipe
    
    train_dataloader_id = DataLoader(
        train_dataset_id,
        batch_size=args.train_batch_size_other,
        shuffle=True,  
        collate_fn=collate_fn, 
        num_workers=args.dataloader_num_workers,
    )
    train_dataloader_motion = DataLoader(
        train_dataset_motion,
        batch_size=args.train_batch_size,
        shuffle=True,  
        collate_fn=collate_fn, 
        num_workers=args.dataloader_num_workers,
    )
    print(highlight_color+"length of id dataloader: ", len(train_dataloader_id))
    print(highlight_color+"length of motion dataloader: ", len(train_dataloader_motion))

    length_dataloader = len(train_dataloader_id)
    length_dataset = len(train_dataset_id)+len(train_dataset_motion)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(length_dataloader / args.gradient_accumulation_steps) 
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    if use_deepspeed_scheduler:
        from accelerate.utils import DummyScheduler

        lr_scheduler = DummyScheduler(
            name=args.lr_scheduler,
            optimizer=optimizer,
            total_num_steps=args.max_train_steps * accelerator.num_processes,
            num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        )
    else:
        lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
            num_training_steps=args.max_train_steps * accelerator.num_processes,
            num_cycles=args.lr_num_cycles,
            power=args.lr_power,
        )

    # Prepare everything with our `accelerator`——对应prepare_for_training
    transformer, optimizer, train_dataloader_id, train_dataloader_motion, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader_id, train_dataloader_motion, lr_scheduler
    )

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_name = args.tracker_name or "cogvideox-lora"
        accelerator.init_trackers(tracker_name, config=vars(args))

    # Train!
    num_trainable_parameters = sum(param.numel() for model in params_to_optimize for param in model["params"])
    
    logger.info("***** Running training *****")
    logger.info(f"  Num trainable parameters = {num_trainable_parameters}")
    logger.info(f"  Num examples = {length_dataset}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Gradient accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    def next_or_restart(iterator, dataloader):
        try:
            return next(iterator), iterator
        except StopIteration:
            iterator = iter(dataloader)
            return next(iterator), iterator
    
    # Potentially load in the weights and states from a previous save
    if not args.resume_from_checkpoint:
        initial_global_step = 0
    else:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    vae_scale_factor_spatial = 2 ** (len(vae.config.block_out_channels) - 1)

    # For DeepSpeed training
    model_config = transformer.module.config if hasattr(transformer, "module") else transformer.config

    # max step trainig 
    iter_id = iter(train_dataloader_id)
    iter_motion = iter(train_dataloader_motion)

    # begin trainig...
    while initial_global_step < args.max_train_steps:
        transformer.train()

        batch_id, iter_id = next_or_restart(iter_id, train_dataloader_id)
        batch_motion, iter_motion = next_or_restart(iter_motion, train_dataloader_motion)
        
        use_motion = random.random() < args.motion_ratio
        batch = batch_motion if use_motion else batch_id
        adapter_flag = ["motion" if use_motion else "id"]

        models_to_accumulate = [transformer]
        with accelerator.accumulate(models_to_accumulate):
            model_input = batch["videos"].permute(0, 2, 1, 3, 4).to(dtype=weight_dtype) # [B, T, C, H, W] -> [1, batch, 16, 60, 90]
            prompts = batch["prompts"]
            print("selected_prompt:", prompts[0])
            print(highlight_color+"adapter_flag:", adapter_flag)

            y_image = None
            condition_flag = False

            check_motion = args.use_MotionAdapter if adapter_flag is None else "motion" in adapter_flag
            check_id = args.use_IdAdapter if adapter_flag is None else "id" in adapter_flag

            if (check_motion and args.use_motion_condition) or (check_id and args.use_id_condition):
                condition_flag = True

            if condition_flag:
                print(highlight_color+"use condition in current module...")
                y_image = batch["vit_frames"].to(dtype=weight_dtype)

            # encode prompts
            prompt_embeds = compute_prompt_embeddings(
                tokenizer,
                text_encoder,
                prompts,
                model_config.max_text_seq_length,
                accelerator.device,
                weight_dtype,
                requires_grad=False,
            )

            # Sample noise that will be added to the latents
            noise = torch.randn_like(model_input)
            batch_size, num_frames, num_channels, height, width = model_input.shape

            # Sample a random timestep for each image
            timesteps = torch.randint(
                    0, scheduler.config.num_train_timesteps, (batch_size,), device=model_input.device
            )

            # Prepare timesteps
            timesteps = timesteps.long()

            # Prepare rotary embeds
            image_rotary_emb = (
                prepare_rotary_positional_embeddings(
                    height=args.height,
                    width=args.width,
                    num_frames=num_frames,
                    vae_scale_factor_spatial=vae_scale_factor_spatial,
                    patch_size=model_config.patch_size,
                    attention_head_dim=model_config.attention_head_dim,
                    device=accelerator.device,
                )
                if model_config.use_rotary_positional_embeddings
                else None
            )

            # Add noise to the model input according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_model_input = scheduler.add_noise(model_input, noise, timesteps)

            # Predict the noise residual
            model_output = transformer(
                hidden_states=noisy_model_input,
                encoder_hidden_states=prompt_embeds,
                timestep=timesteps,
                image_rotary_emb=image_rotary_emb,
                return_dict=False,
                y_image=y_image,
            )[0]
            model_pred = scheduler.get_velocity(model_output, noisy_model_input, timesteps)

            alphas_cumprod = scheduler.alphas_cumprod[timesteps]
            weights = 1 / (1 - alphas_cumprod)
            while len(weights.shape) < len(model_pred.shape):
                weights = weights.unsqueeze(-1)

            target = model_input

            loss = torch.mean((weights * (model_pred - target) ** 2).reshape(batch_size, -1), dim=1)
            loss = loss.mean()
            accelerator.backward(loss)

            if not args.close_grad_mask:
                if "id" in adapter_flag:
                    param_keys = ['motion_attn_adapter', 'motion_ff_adapter']
                elif "motion" in adapter_flag:
                    param_keys = ['id_attn_adapter', 'id_ff_adapter']
                else:
                    raise ValueError(
                        "adapter_flag should be either 'id' or 'motion' when use joint training, but got {}"
                    )
                for name, param in transformer.named_parameters():
                    if any(key in name for key in param_keys):
                        param.grad = None

            if accelerator.sync_gradients:
                params_to_clip = transformer.parameters()
                accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

            if accelerator.state.deepspeed_plugin is None:
                optimizer.step()
                optimizer.zero_grad()

            lr_scheduler.step()

        # Checks if the accelerator has performed an optimization step behind the scenes
        if accelerator.sync_gradients:
            progress_bar.update(1)
            global_step += 1

            if accelerator.is_main_process:
                if global_step % args.checkpointing_steps == 0:
                    # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                    if args.checkpoints_total_limit is not None:
                        checkpoints = os.listdir(args.output_dir)
                        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                        # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                        if len(checkpoints) >= args.checkpoints_total_limit:
                            num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                            removing_checkpoints = checkpoints[0:num_to_remove]

                            logger.info(
                                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                            )
                            logger.info(f"Removing checkpoints: {', '.join(removing_checkpoints)}")

                            for removing_checkpoint in removing_checkpoints:
                                removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                shutil.rmtree(removing_checkpoint)

                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")

        logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
        progress_bar.set_postfix(**logs)
        accelerator.log(logs, step=global_step)

        if accelerator.is_main_process:
            if args.inf_eva_prompts and initial_global_step % args.inference_steps == 0:
                os.makedirs(osp.join(args.output_dir, f'videos'), exist_ok=True)
                inf_transformer = unwrap_model(transformer)
                inf_transformer.eval()

                y_image = None

                if not args.ref_image_path:
                    raise ValueError("requires ref_image_path...")
                
                print(highlight_color +" Loading motion reference image... ")
                y_image = motion_ref_image_encoder(
                    pretrained=args.clip_pretrained, 
                    image_path=args.ref_image_path,
                    device=accelerator.device,
                    dtype=weight_dtype,
                    )

                pipe = load_cogvideox_pipeline(args, weight_dtype, inf_transformer)
                pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
                print(highlight_color+"load successfully")
                
                pipe = pipe.to(accelerator.device)

                prompt_list = load_prompts(args.inf_eva_prompts)
                for i, prompt in enumerate(prompt_list):
                    print(f"\033[1;31m Generating {i+1}/{len(prompt_list)}: {prompt} \033[0m")
                    pipeline_args = {
                        "prompt": prompt,
                        "guidance_scale": 6.0,
                        "use_dynamic_cfg": False,
                        "height": 480,
                        "width": 720,
                        "y_image": y_image,
                    }

                    generator = torch.Generator(device="cuda").manual_seed(42)
                    video = pipe(**pipeline_args, generator=generator, output_type="np").frames[0]
                    print("generate video successful...")
                    # save video
                    current_file_path = os.path.join(osp.join(args.output_dir, f'videos'), f"video_train_{initial_global_step}_case{i+1}.mp4")
                    export_to_video(video, current_file_path, fps=8)
                    print("\033[1;31m saved video to: \033[0m", current_file_path)
    
        initial_global_step += 1
        print("\033[1;31m training step: \033[0m", initial_global_step)

    # Save the adapter layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        transformer = unwrap_model(transformer)
        dtype = (
            torch.float16
            if args.mixed_precision == "fp16"
            else torch.bfloat16
            if args.mixed_precision == "bf16"
            else torch.float32
        )
        transformer = transformer.to(dtype)
        # save adapters
        adapter_path = osp.join(args.output_dir, f'adapter.pth')
        print(f'Begin to Save model to {adapter_path}')

        trainable_dict = {name: param for name, param in transformer.named_parameters() if param.requires_grad}
        torch.save(trainable_dict, adapter_path)
        print(f'Save model to {adapter_path}')

    accelerator.end_training() 
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    args = get_args()
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    args_dict = vars(args)
    args_file_path = os.path.join(args.output_dir, "training_args.json")
    with open(args_file_path, 'w') as f:
        json.dump(args_dict, f, indent=4)

    main(args)