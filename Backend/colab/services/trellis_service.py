import os
import gc
from PIL import Image
from typing import Optional

class TrellisService:
    def __init__(self):
        self.pipeline = None
        self.initialized = False

    def initialize_model(self):
        """
        Lazy loads the Microsoft TRELLIS pipeline and weights from Hugging Face.
        Requires a CUDA-compatible GPU.
        """
        if self.initialized:
            return

        print(">>> Initializing Microsoft TRELLIS Image-to-3D Pipeline...")
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. TRELLIS requires a CUDA-enabled GPU.")

        os.environ['SPCONV_ALGO'] = 'native'
        
        # Load the SOTA TRELLIS pipeline
        from trellis.pipelines import TrellisImageTo3DPipeline
        self.pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
        self.pipeline.cuda()
        self.initialized = True
        print("--- TRELLIS Pipeline successfully loaded on GPU ---")

    def generate_mesh(self, input_image_path: str, output_glb_path: str, remove_bg: bool = True) -> bool:
        """
        Runs background removal and TRELLIS inference to generate a 3D GLB model.
        """
        try:
            # 1. Ensure model is loaded
            self.initialize_model()
            
            import torch
            from rembg import remove
            from trellis.utils import postprocessing_utils
            
            # 2. Open input image
            print(f"Loading input image: {input_image_path}")
            img = Image.open(input_image_path)
            
            # 3. Perform background removal if requested (TRELLIS works best with transparent backgrounds)
            if remove_bg:
                print("Removing background using rembg...")
                img = remove(img)
                # Ensure transparent pixels are treated properly (convert to RGBA)
                img = img.convert("RGBA")
                
            # 4. Run TRELLIS pipeline
            print("Running TRELLIS model inference (this may take 10-15 seconds)...")
            with torch.no_grad():
                outputs = self.pipeline.run(img, seed=42)
                
            # 5. Extract mesh and post-process to GLB
            print("Post-processing 3D Gaussian and Mesh data to GLB container...")
            
            # Extract first batch items
            gaussian_item = outputs['gaussian'][0]
            mesh_item = outputs['mesh'][0]
            
            # Convert to standard GLB with textures
            glb = postprocessing_utils.to_glb(
                gaussian_item,
                mesh_item,
                simplify=0.90,       # Decimates triangles slightly for web optimization
                texture_size=1024     # High-resolution texture maps
            )
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_glb_path), exist_ok=True)
            
            # Save the GLB file
            glb.export(output_glb_path)
            print(f"--- Photorealistic GLB compiled successfully: {output_glb_path} ---")
            
            # 6. Garbage collection to prevent CUDA OOM
            del outputs, glb
            gc.collect()
            torch.cuda.empty_cache()
            
            return True
            
        except Exception as e:
            print(f"[ERROR] TRELLIS mesh generation pipeline failure: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

# Global instance of TrellisService
trellis_service = TrellisService()
