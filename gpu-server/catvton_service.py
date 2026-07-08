import os
import sys
import torch
import numpy as np
from PIL import Image

# Add CatVTON path dynamically to load model components
current_dir = os.path.dirname(os.path.abspath(__file__))
catvton_candidates = [
    os.environ.get("CATVTON_ROOT"),
    os.path.abspath(os.path.join(current_dir, "..", "models", "CatVTON")),
    os.path.abspath(os.path.join(current_dir, "models", "CatVTON")),
    "/app/models/CatVTON"
]

catvton_dir = None
for candidate in catvton_candidates:
    if candidate and os.path.exists(candidate):
        catvton_dir = candidate
        break

if catvton_dir:
    print(f"[CATVTON] Loading module from: {catvton_dir}")
    sys.path.insert(0, catvton_dir)
else:
    raise FileNotFoundError(
        "CatVTON repository directory not found. Please clone it to 'models/CatVTON' or define 'CATVTON_ROOT' environment variable."
    )

from model.pipeline import CatVTONPipeline
from model.cloth_masker import AutoMasker
from utils import init_weight_dtype, resize_and_crop, resize_and_padding

class CatVtonService:
    def __init__(self):
        self.pipeline = None
        self.automasker = None
        self.mask_processor = None
        self.initialized = False

    def initialize(self):
        if self.initialized:
            return
        
        from huggingface_hub import snapshot_download
        from diffusers.image_processor import VaeImageProcessor
        
        print(">>> Downloading/Loading CatVTON weights...")
        repo_path = snapshot_download(repo_id="zhengchong/CatVTON")
        print(f"CatVTON weights located at: {repo_path}")
        
        print(">>> Initializing CatVTON Pipeline on GPU...")
        self.pipeline = CatVTONPipeline(
            base_ckpt="runwayml/stable-diffusion-inpainting",
            attn_ckpt=repo_path,
            attn_ckpt_version="mix",
            weight_dtype=torch.bfloat16,
            use_tf32=True,
            device='cuda',
            skip_safety_check=True
        )
        
        self.mask_processor = VaeImageProcessor(vae_scale_factor=8, do_normalize=False, do_binarize=True, do_convert_grayscale=True)
        
        print(">>> Initializing AutoMasker (DensePose & SCHP)...")
        self.automasker = AutoMasker(
            densepose_ckpt=os.path.join(repo_path, "DensePose"),
            schp_ckpt=os.path.join(repo_path, "SCHP"),
            device='cuda', 
        )
        
        self.initialized = True
        print(">>> CatVTON Service successfully initialized!")

    def run_tryon(self, person_path: str, garment_path: str, cloth_type: str = "upper", seed: int = -1, steps: int = 50, cfg: float = 2.5) -> Image.Image:
        self.initialize()
        
        # Load and convert images to RGB
        person_image = Image.open(person_path).convert("RGB")
        cloth_image = Image.open(garment_path).convert("RGB")
        
        width, height = 768, 1024
        person_image = resize_and_crop(person_image, (width, height))
        cloth_image = resize_and_padding(cloth_image, (width, height))
        
        # Run AutoMasker to generate mask
        print(f"Generating garment mask for cloth type: {cloth_type}...")
        mask = self.automasker(person_image, cloth_type)['mask']
        mask = self.mask_processor.blur(mask, blur_factor=9)
        
        # Generator seed
        generator = None
        if seed != -1:
            generator = torch.Generator(device='cuda').manual_seed(seed)
            
        print("Running CatVTON diffusion try-on model...")
        result_image = self.pipeline(
            image=person_image,
            condition_image=cloth_image,
            mask=mask,
            num_inference_steps=steps,
            guidance_scale=cfg,
            generator=generator
        )[0]
        
        return result_image

    def unload_models(self):
        """
        Unloads CatVTON model components from GPU memory and clears the CUDA cache.
        """
        if not self.initialized:
            return
        print(">>> [VRAM] Unloading CatVTON pipeline & AutoMasker from GPU...")
        self.pipeline = None
        self.automasker = None
        self.mask_processor = None
        self.initialized = False
        
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(">>> [VRAM] CatVTON VRAM released successfully.")

catvton_service = CatVtonService()
