# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
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
from basicsr.utils import img2tensor, tensor2img
from facelib.utils.face_restoration_helper import FaceRestoreHelper
from codeformer.archs.codeformer_arch import CodeFormer

# ✅ Load Super-Resolution Model Properly
def load_superresolution_model(method):
    """Loads the appropriate super-resolution model (GFPGAN or CodeFormer)."""
    if method == "GFPGAN":
        from gfpgan import GFPGANer
        return GFPGANer(model_path="checkpoints/gfpgan/GFPGANv1.4.pth", upscale=2, arch="clean", channel_multiplier=2).to("cuda").eval()
    elif method == "CodeFormer":
        from facelib.utils import CodeFormer
        return CodeFormer(model_path="checkpoints/codeformer/codeformer.pth", upscale=2).to("cuda").eval()
    return None

# ✅ Apply Super-Resolution on Each Frame
def apply_superresolution(frame, model, method):
    """Applies super-resolution to the entire frame using the selected model."""
    img_tensor = img2tensor(frame, bgr2rgb=True, float32=True) / 255.0
    img_tensor = img_tensor.unsqueeze(0).to("cuda")  # Add batch dimension

    if method == "GFPGAN":
        _, enhanced_img = model.enhance(img_tensor, has_aligned=True, only_center_face=False)
    elif method == "CodeFormer":
        enhanced_img = model(img_tensor)
    else:
        return frame

    return tensor2img(enhanced_img, rgb2bgr=True)

# ✅ Enhance Video Frames with Super-Resolution
def enhance_generated_frames(video_out_path, superres_method):
    """Applies super-resolution enhancement to all frames in the video."""
    cap = cv2.VideoCapture(video_out_path)
    if not cap.isOpened():
        raise RuntimeError(f"Error opening video file: {video_out_path}")

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    output_path = video_out_path.replace(".mp4", "_enhanced.mp4")
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))

    model = load_superresolution_model(superres_method)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        enhanced_frame = apply_superresolution(frame, model, superres_method)
        out.write(enhanced_frame)

    cap.release()
    out.release()
    print(f"Super-resolution completed. Output saved at: {output_path}")

# ✅ Main Function
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

    # ✅ Apply Super-Resolution If Selected
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
