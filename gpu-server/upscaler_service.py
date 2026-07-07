import os
import sys
import torch
import urllib.request
import numpy as np
from PIL import Image

# URLs for model weights
REALESRGAN_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
GFPGAN_URL = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"

class UpscalerService:
    def __init__(self):
        self.initialized = False
        self.fallback = False
        self.upscaler = None
        self.face_restorer = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Sibling models directory next to server folder (keeps code lightweight)
        server_dir = os.path.dirname(os.path.abspath(__file__))
        sibling_models_dir = os.path.abspath(os.path.join(server_dir, "..", "models"))
        
        if os.path.exists(sibling_models_dir):
            self.weights_dir = os.path.join(sibling_models_dir, "upscaler")
        else:
            self.weights_dir = os.path.join(server_dir, "weights")

        self.realesrgan_path = os.path.join(self.weights_dir, "RealESRGAN_x4plus.pth")
        self.gfpgan_path = os.path.join(self.weights_dir, "GFPGANv1.4.pth")

        # Test imports to check if dependencies are available
        try:
            import realesrgan
            import gfpgan
            import basicsr
            print(">>> [UPSCALER] GFPGAN and Real-ESRGAN dependencies imported successfully.")
        except ImportError as e:
            print(f">>> [UPSCALER WARNING] Missing dependencies: {str(e)}")
            print(">>> [UPSCALER] Running in Fallback Mode (PIL Lanczos interpolation will be used instead of AI).")
            self.fallback = True

    def download_weights(self):
        """
        Downloads Real-ESRGAN and GFPGAN model weights if they are not already cached.
        """
        os.makedirs(self.weights_dir, exist_ok=True)
        
        # Download Real-ESRGAN
        if not os.path.exists(self.realesrgan_path):
            print(f">>> [UPSCALER] Downloading Real-ESRGAN weights to {self.realesrgan_path}...")
            try:
                urllib.request.urlretrieve(REALESRGAN_URL, self.realesrgan_path)
                print(">>> [UPSCALER] Real-ESRGAN weights downloaded successfully.")
            except Exception as err:
                print(f"[DOWNLOAD ERROR] Failed to retrieve Real-ESRGAN weights: {str(err)}")
                raise err

        # Download GFPGAN
        if not os.path.exists(self.gfpgan_path):
            print(f">>> [UPSCALER] Downloading GFPGAN weights to {self.gfpgan_path}...")
            try:
                urllib.request.urlretrieve(GFPGAN_URL, self.gfpgan_path)
                print(">>> [UPSCALER] GFPGAN weights downloaded successfully.")
            except Exception as err:
                print(f"[DOWNLOAD ERROR] Failed to retrieve GFPGAN weights: {str(err)}")
                raise err

    def initialize(self):
        if self.initialized or self.fallback:
            return

        try:
            self.download_weights()

            from realesrgan import RealESRGANer
            from basicsr.archs.rrdbnet_arch import RRDBNet
            try:
                from gfpgan import GFPGANer
            except ImportError:
                from gfpgan.utils import GFPGANer

            print(f">>> [UPSCALER] Initializing AI models on device: {self.device}...")

            # 1. Initialize Real-ESRGAN background upscaler
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            self.upscaler = RealESRGANer(
                scale=4,
                model_path=self.realesrgan_path,
                model=model,
                tile=400,
                tile_pad=10,
                pre_pad=0,
                half=(self.device == 'cuda'), # FP16 is only stable on CUDA
                device=self.device
            )

            # 2. Initialize GFPGAN face restorer with background upsampler integrated
            self.face_restorer = GFPGANer(
                model_path=self.gfpgan_path,
                upscale=4,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=self.upscaler,
                device=self.device
            )

            self.initialized = True
            print(">>> [UPSCALER] AI Upscaling & Face Restoration service successfully warmed up!")
        except Exception as e:
            print(f">>> [UPSCALER ERROR] Failed to initialize AI models: {str(e)}")
            print(">>> [UPSCALER] Falling back to PIL Lanczos upscaling.")
            self.fallback = True

    def enhance_image(self, pil_image: Image.Image) -> Image.Image:
        """
        Upscales input PIL Image by 4x and restores facial details.
        """
        # Ensure service is initialized
        self.initialize()

        if self.fallback:
            # High quality bicubic fallback if libraries are missing or failed to initialize
            print(">>> [UPSCALER] Enhancing image using PIL Lanczos 4x interpolation...")
            w, h = pil_image.size
            return pil_image.resize((w * 4, h * 4), Image.Resampling.LANCZOS)

        try:
            print(">>> [UPSCALER] Running AI Super-Resolution & Face Restoration...")
            # Convert PIL Image to OpenCV BGR format
            cv_img = np.array(pil_image.convert("RGB"))
            cv_img = cv_img[:, :, ::-1] # RGB to BGR

            # Run GFPGAN (which triggers Real-ESRGAN internally on background/clothes)
            _, _, restored_img = self.face_restorer.enhance(
                cv_img,
                has_aligned=False,
                only_center_face=False,
                paste_back=True
            )

            # Convert BGR back to RGB and then to PIL Image
            restored_rgb = restored_img[:, :, ::-1]
            return Image.fromarray(restored_rgb)
        except Exception as err:
            print(f">>> [UPSCALER ERROR] AI enhancement failed: {str(err)}")
            print(">>> [UPSCALER] Executing fallback PIL Lanczos upscaling.")
            w, h = pil_image.size
            return pil_image.resize((w * 4, h * 4), Image.Resampling.LANCZOS)

    def unload_models(self):
        """
        Unloads Real-ESRGAN and GFPGAN components from GPU memory and clears the CUDA cache.
        """
        if not self.initialized:
            return
        print(">>> [VRAM] Unloading AI Super-Resolution & Face Restoration models from GPU...")
        self.upscaler = None
        self.face_restorer = None
        self.initialized = False
        
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(">>> [VRAM] Super-Resolution VRAM released successfully.")

upscaler_service = UpscalerService()
