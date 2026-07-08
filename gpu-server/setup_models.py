import os
import sys
import urllib.request
import tarfile
import shutil
import subprocess

def print_banner(text):
    print("\n" + "=" * 60)
    print(f" {text}")
    print("=" * 60)

def download_file(url, dest_path):
    if os.path.exists(dest_path):
        print(f"✅ Already exists: {os.path.basename(dest_path)}")
        return True
    
    print(f"📥 Downloading {url} to {dest_path}...")
    try:
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        # Using a custom opener to support User-Agent header (prevents HTTP 403 on some servers)
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)')]
        urllib.request.install_opener(opener)
        
        urllib.request.urlretrieve(url, dest_path)
        print(f"✅ Successfully downloaded: {os.path.basename(dest_path)}")
        return True
    except Exception as e:
        print(f"❌ Failed to download {url}: {str(e)}")
        return False

def main():
    # Set environment variables to increase Hugging Face Hub timeouts and stability
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_ETAG_TIMEOUT"] = "60"
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "60"

    print_banner("VR Try-On Studio: AI Models Weights Downloader")
    
    gpu_server_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(gpu_server_dir, ".."))
    
    # 1. HuggingFace Hub downloads (CatVTON & Stable Diffusion)
    print_banner("1. Downloading Hugging Face Models (CatVTON & SD Inpainting)")
    try:
        from huggingface_hub import snapshot_download
        import torch
        from diffusers import StableDiffusionInpaintPipeline
        
        print("📥 Downloading CatVTON model weights...")
        catvton_path = snapshot_download(repo_id="zhengchong/CatVTON")
        print(f"✅ CatVTON weights loaded to: {catvton_path}")
        
        print("📥 Downloading runwayml/stable-diffusion-inpainting (bfloat16 subset)...")
        # Using from_pretrained with torch_dtype is much smarter than downloading the whole snapshot:
        # It skips the massive 4.27 GB original .ckpt weights and other redundant formats,
        # downloading only the exact files needed for inference.
        StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True
        )
        print("✅ Stable Diffusion Inpainting weights successfully downloaded.")
    except Exception as e:
        print(f"❌ Failed HF Hub download: {str(e)}")
        print("Make sure 'huggingface-hub' and 'diffusers' are installed and you have an active internet connection.")

    # 2. AI Upscaling & Face Restoration weights
    print_banner("2. Downloading AI Upscaler & Face Restoration Models")
    upscaler_weights_dir = os.path.join(project_root, "models", "upscaler")
    if not os.path.exists(upscaler_weights_dir):
        upscaler_weights_dir = os.path.join(gpu_server_dir, "weights")
        
    realesrgan_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
    gfpgan_url = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
    
    download_file(realesrgan_url, os.path.join(upscaler_weights_dir, "RealESRGAN_x4plus.pth"))
    download_file(gfpgan_url, os.path.join(upscaler_weights_dir, "GFPGANv1.4.pth"))

    # 3. 4D-Humans (HMR2) & Detectron2 weights
    print_banner("3. Downloading 4D-Humans (HMR2) & Detectron2 Bounding Box Model")
    home_dir = os.environ.get("HOME") or os.environ.get("USERPROFILE") or os.path.expanduser("~")
    cache_dir_4dhumans = os.path.join(home_dir, ".cache", "4DHumans")
    
    # Download HMR2 Data tarball
    hmr2_tar_url = "https://www.cs.utexas.edu/~pavlakos/4dhumans/hmr2_data.tar.gz"
    tar_dest = os.path.join(cache_dir_4dhumans, "hmr2_data.tar.gz")
    
    if download_file(hmr2_tar_url, tar_dest):
        # Extract if hmr2_data folder is not already there
        checkpoints_dir = os.path.join(cache_dir_4dhumans, "logs", "train", "multiruns", "hmr2", "0", "checkpoints")
        if not os.path.exists(checkpoints_dir):
            print("📦 Extracting hmr2_data.tar.gz...")
            try:
                with tarfile.open(tar_dest, "r:gz") as tar:
                    tar.extractall(path=cache_dir_4dhumans)
                print("✅ Extraction complete.")
            except Exception as e:
                print(f"❌ Failed to extract HMR2 data: {str(e)}")
        else:
            print("✅ HMR2 checkpoints already extracted.")

    # Download Detectron2 ViTDet COCO weights
    vitdet_url = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
    # Detectron2/fvcore cache defaults to ~/.torch/fvcore_cache/detectron2/ViTDet/... or similar
    fvcore_cache_dir = os.path.join(home_dir, ".torch", "fvcore_cache", "detectron2", "ViTDet", "COCO", "cascade_mask_rcnn_vitdet_h", "f328730692")
    download_file(vitdet_url, os.path.join(fvcore_cache_dir, "model_final_f05665.pkl"))

    # 4. LHM++ prior and model weights
    print_banner("4. Downloading LHM++ (Large Human Model ++) Assets")
    sibling_lhm_root = os.path.abspath(os.path.join(project_root, "models", "LHM-plusplus"))
    lhm_root = os.environ.get("LHM_ROOT")
    if not lhm_root:
        lhm_root = sibling_lhm_root
    
    if os.path.exists(lhm_root):
        print(f"LHM++ repository found at: {lhm_root}")
        
        # Download prior models via LHM++ script
        lhm_download_script = os.path.join(lhm_root, "scripts", "download_pretrained_models.py")
        if os.path.exists(lhm_download_script):
            print("📥 Running LHM++ prior downloader script...")
            try:
                subprocess.run([sys.executable, lhm_download_script, "--prior"], cwd=lhm_root, check=True)
                print("✅ LHM++ prior models successfully configured.")
            except Exception as e:
                print(f"⚠️ Warning: LHM++ prior downloader script failed: {str(e)}")
        else:
            print(f"⚠️ Warning: download_pretrained_models.py not found at {lhm_download_script}")
            
        # Download specific LHMPP-700M-PixelShuffle model using ModelScope or HF
        print("📥 Downloading LHMPP-700M-PixelShuffle weights...")
        lhm_weights_dir = os.path.join(lhm_root, "pretrained_models", "Damo_XR_Lab", "LHMPP-700M-PixelShuffle")
        if not os.path.exists(lhm_weights_dir):
            # Try ModelScope snapshot download first
            try:
                from modelscope import snapshot_download
                print("Using ModelScope to download LHMPP-700M-PixelShuffle...")
                snapshot_download('Damo_XR_Lab/LHMPP-700M-PixelShuffle', cache_dir=os.path.join(lhm_root, 'pretrained_models'))
                print("✅ LHMPP-700M-PixelShuffle model successfully downloaded.")
            except ImportError:
                # If ModelScope isn't installed, check if huggingface_hub can download it (3DAIGC/LHMPP-700M-PixelShuffle)
                try:
                    from huggingface_hub import snapshot_download
                    print("Using Hugging Face to download LHMPP-700M-PixelShuffle...")
                    local_dir = os.path.join(lhm_root, "pretrained_models", "Damo_XR_Lab", "LHMPP-700M-PixelShuffle")
                    snapshot_download(repo_id="3DAIGC/LHMPP-700M-PixelShuffle", local_dir=local_dir)
                    print("✅ LHMPP-700M-PixelShuffle model successfully downloaded.")
                except Exception as e:
                    print(f"❌ Failed to download LHMPP-700M-PixelShuffle: {str(e)}")
                    print("Please install modelscope ('pip install modelscope') or huggingface-hub to auto-download this.")
        else:
            print("✅ LHMPP-700M-PixelShuffle model weights already exist.")
    else:
        print(f"ℹ️ LHM++ project directory not found at '{lhm_root}'. Skipping LHM++ model setup.")
        print("To configure LHM++ model downloads, define the 'LHM_ROOT' environment variable pointing to the LHM-plusplus directory.")

    print_banner("PRE-DOWNLOAD SETUP COMPLETED")
    print("All configured AI models and weights are now cached and ready for runtime execution!")

if __name__ == "__main__":
    main()
