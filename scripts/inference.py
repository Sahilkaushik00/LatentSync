# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
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

import argparse
from omegaconf import OmegaConf
import torch
import cv2
import numpy as np
from diffusers import AutoencoderKL, DDIMScheduler
from latentsync.models.unet import UNet3DConditionModel
from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
from diffusers.utils.import_utils import is_xformers_available
from accelerate.utils import set_seed
from latentsync.whisper.audio2feature import Audio2Feature
from basicsr.archs.gfpganv1_clean_arch import GFPGANv1Clean
from basicsr.utils import img2tensor, tensor2img
from facelib.utils.face_restoration_helper import FaceRestoreHelper
from codeformer.archs.codeformer_arch import CodeFormer

def apply_superresolution(image, method="GFPGAN"):
    """Applies super-resolution enhancement using GFPGAN or CodeFormer."""
    if method == "GFPGAN":
        gfpgan = GFPGANv1Clean(arch="clean", channel_multiplier=2)
        gfpgan.load_weights("checkpoints/GFPGANv1.pth", strict=True)
        gfpgan.eval()
        img_tensor = img2tensor(image, bgr2rgb=True, float32=True) / 255.0
        _, restored_img = gfpgan(img_tensor)
        return tensor2img(restored_img, rgb2bgr=True)

    elif method == "CodeFormer":
        codeformer = CodeFormer()
        codeformer.load_weights("checkpoints/CodeFormer.pth", strict=True)
        codeformer.eval()
        img_tensor = img2tensor(image, bgr2rgb=True, float32=True) / 255.0
        restored_img = codeformer(img_tensor)
        return tensor2img(restored_img, rgb2bgr=True)

    return image  # Return original if no method is specified

def enhance_generated_frames(video_out_path, superres_method):
    """Enhances only the generated parts of the frames using super-resolution."""
    cap = cv2.VideoCapture(video_out_path)
    if not cap.isOpened():
        raise RuntimeError(f"Error opening video file: {video_out_path}")

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    out = cv2.VideoWriter(video_out_path.replace(".mp4", "_enhanced.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Simulated logic to detect generated region (replace with actual mask detection logic)
        generated_region = frame[frame_height//4: 3*frame_height//4, frame_width//4: 3*frame_width//4]
        
        # Check if resolution is degraded (compare with a reference, here assumed)
        if np.mean(generated_region) < 150:  # Example threshold, replace with actual check
            enhanced_region = apply_superresolution(generated_region, method=superres_method)
            frame[frame_height//4: 3*frame_height//4, frame_width//4: 3*frame_width//4] = enhanced_region

        out.write(frame)

    cap.release()
    out.release()

def main(config, args):
    is_fp16_supported = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] > 7
    dtype = torch.float16 if is_fp16_supported else torch.float32

    print(f"Input video path: {args.video_path}")
    print(f"Input audio path: {args.audio_path}")
    print(f"Loaded checkpoint path: {args.inference_ckpt_path}")

    scheduler = DDIMScheduler.from_pretrained("configs")

    if config.model.cross_attention_dim == 768:
        whisper_model_path = "checkpoints/whisper/small.pt"
    elif config.model.cross_attention_dim == 384:
        whisper_model_path = "checkpoints/whisper/tiny.pt"
    else:
        raise NotImplementedError("cross_attention_dim must be 768 or 384")

    audio_encoder = Audio2Feature(model_path=whisper_model_path, device="cuda", num_frames=config.data.num_frames)

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=dtype)
    vae.config.scaling_factor = 0.18215
    vae.config.shift_factor = 0

    unet, _ = UNet3DConditionModel.from_pretrained(
        OmegaConf.to_container(config.model),
        args.inference_ckpt_path,
        device="cpu",
    )

    unet = unet.to(dtype=dtype)

    if is_xformers_available():
        unet.enable_xformers_memory_efficient_attention()

    pipeline = LipsyncPipeline(
        vae=vae,
        audio_encoder=audio_encoder,
        unet=unet,
        scheduler=scheduler,
    ).to("cuda")

    if args.seed != -1:
        set_seed(args.seed)
    else:
        torch.seed()

    print(f"Initial seed: {torch.initial_seed()}")

    pipeline(
        video_path=args.video_path,
        audio_path=args.audio_path,
        video_out_path=args.video_out_path,
        video_mask_path=args.video_out_path.replace(".mp4", "_mask.mp4"),
        num_frames=config.data.num_frames,
        num_inference_steps=args.inference_steps,
        guidance_scale=args.guidance_scale,
        weight_dtype=dtype,
        width=config.data.resolution,
        height=config.data.resolution,
    )

    # Apply super-resolution if selected
    if args.superres and args.superres in ["GFPGAN", "CodeFormer"]:
        print(f"Applying super-resolution with {args.superres}...")
        enhance_generated_frames(args.video_out_path, args.superres)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--unet_config_path", type=str, default="configs/unet.yaml")
    parser.add_argument("--inference_ckpt_path", type=str, required=True)
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--audio_path", type=str, required=True)
    parser.add_argument("--video_out_path", type=str, required=True)
    parser.add_argument("--inference_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1247)
    parser.add_argument("--superres", type=str, choices=["GFPGAN", "CodeFormer"], help="Select super-resolution method")

    args = parser.parse_args()

    config = OmegaConf.load(args.unet_config_path)

    main(config, args)
