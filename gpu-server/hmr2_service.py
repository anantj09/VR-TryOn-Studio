import os
import sys
import torch
import numpy as np
import cv2
import trimesh
from pathlib import Path
from typing import Tuple

# Add the parent directory of hmr2 to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class Hmr2Service:
    def __init__(self):
        self.initialized = False
        self.model = None
        self.model_cfg = None
        self.detector = None
        self.device = None

    def initialize(self):
        if self.initialized:
            return
            
        # Set up numpy compatibility patches for older library calls
        np.object = object
        np.str = str
        np.unicode = str

        from hmr2.configs import CACHE_DIR_4DHUMANS
        from hmr2.models import download_models, load_hmr2, DEFAULT_CHECKPOINT
        
        # 1. Download HMR2 model weights (saves checkpoints to ~/.cache/4DHumans)
        try:
            download_models(CACHE_DIR_4DHUMANS)
        except Exception as e:
            print(f"[WARNING] 4D-Humans weights download failed: {str(e)}")
            
        print(">>> Initializing HMR 2.0 network layers on GPU...")
        self.model, self.model_cfg = load_hmr2(DEFAULT_CHECKPOINT)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # 2. Initialize Detectron2 human bbox detector
        print(">>> Initializing Detectron2 human bounding-box detector...")
        from hmr2.utils.utils_detectron2 import DefaultPredictor_Lazy
        from detectron2.config import LazyConfig
        import hmr2
        
        cfg_path = Path(hmr2.__file__).parent / 'configs' / 'cascade_mask_rcnn_vitdet_h_75ep.py'
        detectron2_cfg = LazyConfig.load(str(cfg_path))
        # Point to standard COCO detector weights
        detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
            
        self.detector = DefaultPredictor_Lazy(detectron2_cfg)
        self.initialized = True
        print(f"--- 4D-Humans successfully mapped to hardware: [{self.device.type.upper()}] ---")

    def unload_models(self):
        """
        Unloads 4D-Humans and Detectron2 models from GPU memory to reclaim VRAM.
        """
        if not self.initialized:
            return
        print(">>> [VRAM] Unloading 4D-Humans & Detectron2 from GPU...")
        self.model = None
        self.detector = None
        self.initialized = False
        
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(">>> [VRAM] 4D-Humans VRAM released successfully.")

    def generate_hmr2_mesh(self, person_img_path: str, output_path: str, landmarks: list = None) -> bool:
        """
        Runs vitdet and 4D-Humans inference on the GPU server, textures backfaces adaptively,
        and saves a compiled GLB mesh.
        """
        try:
            self.initialize()
            
            # Read input image
            img_cv2 = cv2.imread(person_img_path)
            if img_cv2 is None:
                print(f"Error: Unable to read image path: {person_img_path}")
                return False
                
            h_img, w_img, _ = img_cv2.shape
            
            # Map 2D landmarks to construct bounding box
            if landmarks:
                xs = [pt["x"] * w_img for pt in landmarks if isinstance(pt, dict) and "x" in pt]
                ys = [pt["y"] * h_img for pt in landmarks if isinstance(pt, dict) and "y" in pt]
                
                if xs and ys:
                    min_x, max_x = min(xs), max(xs)
                    min_y, max_y = min(ys), max(ys)
                    w = max_x - min_x
                    h = max_y - min_y
                    pad_x = w * 0.15
                    pad_y = h * 0.15
                    box = np.array([[
                        max(0.0, min_x - pad_x),
                        max(0.0, min_y - pad_y),
                        min(float(w_img), max_x + pad_x),
                        min(float(h_img), max_y + pad_y)
                    ]], dtype=np.float32)
                else:
                    box = None
            else:
                box = None

            if box is None:
                # Fallback to Detectron2 bbox detector
                det_out = self.detector(img_cv2)
                det_instances = det_out['instances']
                valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
                boxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
                if len(boxes) > 0:
                    box = boxes[0:1]
                else:
                    box = np.array([[0.0, 0.0, float(w_img), float(h_img)]], dtype=np.float32)

            # Run 4D-Humans estimator internally to perform 3D human pose recovery
            from hmr2.datasets.vitdet_dataset import ViTDetDataset
            from hmr2.utils import recursive_to
            from hmr2.utils.renderer import cam_crop_to_full
            
            dataset = ViTDetDataset(self.model_cfg, img_cv2, box)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
            batch = next(iter(dataloader))
            batch = recursive_to(batch, self.device)
            
            with torch.no_grad():
                out = self.model(batch)
                
            pred_vertices = out['pred_vertices'][0].detach().cpu().numpy()
            pred_cam = out['pred_cam']
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            
            scaled_focal_length = self.model_cfg.EXTRA.FOCAL_LENGTH / self.model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length)
            cam_t = pred_cam_t_full[0].detach().cpu().numpy()
            
            # Project 3D vertices onto 2D image coordinates to sample colors
            img_size_max = max(h_img, w_img)
            f = (self.model_cfg.EXTRA.FOCAL_LENGTH / self.model_cfg.MODEL.IMAGE_SIZE) * img_size_max
            
            vertices_cam = pred_vertices + cam_t
            cx = w_img / 2.0
            cy = h_img / 2.0
            
            X = vertices_cam[:, 0]
            Y = vertices_cam[:, 1]
            Z = np.where(vertices_cam[:, 2] == 0, 1e-5, vertices_cam[:, 2])
            
            x_proj = f * (X / Z) + cx
            y_proj = f * (Y / Z) + cy
            
            cols = np.clip(np.round(x_proj), 0, w_img - 1).astype(np.int32)
            rows = np.clip(np.round(y_proj), 0, h_img - 1).astype(np.int32)
            
            colors_list = []
            for r, c in zip(rows, cols):
                bgr = img_cv2[r, c]
                rgb = [bgr[2] / 255.0, bgr[1] / 255.0, bgr[0] / 255.0, 1.0]
                colors_list.append(rgb)
            vertex_colors = np.array(colors_list)
            
            # Compile Rigged Mesh
            print("Compiling mesh with sampled vertex colors...")
            faces = self.model.smpl.faces
            if hasattr(faces, 'cpu'):
                faces = faces.cpu().numpy()
                
            mesh = trimesh.Trimesh(pred_vertices, faces.copy(), vertex_colors=vertex_colors)
            
            # Apply rotations to align coordinate axes to OpenGL/Filament standard
            rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
            mesh.apply_transform(rot)
            
            normals = mesh.vertex_normals
            vertices = mesh.vertices
            vertex_colors = mesh.visual.vertex_colors / 255.0  # Convert to float RGBA [0..1]
            
            # --- Phase 1: Collect front-facing vertex data ---
            front_data = []
            for i in range(len(vertices)):
                nz = normals[i][2]
                color = vertex_colors[i]
                is_background = color[0] > 0.90 and color[1] > 0.90 and color[2] > 0.90
                if nz > 0.3 and not is_background:
                    front_data.append((vertices[i][1], vertices[i][0], color, i))

            # --- Phase 2: Detect boundaries ---
            front_data.sort(key=lambda x: -x[0])
            y_min_body = min(v[1] for v in vertices)
            y_max_body = max(v[1] for v in vertices)
            y_range = y_max_body - y_min_body
            
            num_slices = 50
            slice_colors = {}
            for fd in front_data:
                y_norm = (fd[0] - y_min_body) / max(y_range, 0.001)
                slice_idx = min(int(y_norm * num_slices), num_slices - 1)
                if slice_idx not in slice_colors:
                    slice_colors[slice_idx] = []
                if abs(fd[1]) < 0.14:
                    slice_colors[slice_idx].append(fd[2][:3])

            slice_avg = {}
            for idx in range(num_slices):
                if idx in slice_colors and len(slice_colors[idx]) >= 3:
                    slice_avg[idx] = np.mean(slice_colors[idx], axis=0)

            def color_distance(c1, c2):
                return np.sqrt(np.sum((c1 - c2) ** 2))

            boundaries = []
            sorted_slices = sorted(slice_avg.keys())
            for k in range(len(sorted_slices) - 1):
                s1 = sorted_slices[k]
                s2 = sorted_slices[k + 1]
                dist = color_distance(slice_avg[s1], slice_avg[s2])
                if dist > 0.12:
                    boundary_y_norm = (s1 + s2) / 2.0 / num_slices
                    boundaries.append((boundary_y_norm, dist))

            boundaries.sort(key=lambda x: x[0])
            default_boundaries = {
                "shoes_top": 0.08,
                "pants_top": 0.45,
                "tshirt_top": 0.78,
                "skin_top": 0.85
            }

            detected = default_boundaries.copy()
            if len(boundaries) >= 2:
                strongest = sorted(boundaries, key=lambda x: -x[1])[:3]
                strongest_y = sorted([b[0] for b in strongest])
                if len(strongest_y) >= 2:
                    detected["pants_top"] = strongest_y[0]
                    detected["tshirt_top"] = strongest_y[1]
                if len(strongest_y) >= 3:
                    detected["skin_top"] = strongest_y[2]

            boundary_y = {
                "shoes_top": y_min_body + detected["shoes_top"] * y_range,
                "pants_top": y_min_body + detected["pants_top"] * y_range,
                "tshirt_top": y_min_body + detected["tshirt_top"] * y_range,
                "skin_top": y_min_body + detected["skin_top"] * y_range,
            }

            # --- Phase 3: Segment classification ---
            def get_segment_name(v):
                y = v[1]
                x_abs = abs(v[0])
                if y > boundary_y["skin_top"]:
                    return "hair"
                elif y > boundary_y["tshirt_top"]:
                    return "skin"
                elif y > boundary_y["pants_top"]:
                    if x_abs > 0.18:
                        return "skin"
                    return "tshirt"
                elif y > boundary_y["shoes_top"]:
                    return "pants"
                else:
                    return "shoes"

            segment_fallbacks = {
                "hair": np.array([0.18, 0.14, 0.12, 1.0]),
                "skin": np.array([0.85, 0.68, 0.58, 1.0]),
                "tshirt": np.array([0.24, 0.24, 0.25, 1.0]),
                "pants": np.array([0.62, 0.44, 0.28, 1.0]),
                "shoes": np.array([0.35, 0.22, 0.15, 1.0])
            }

            # --- Phase 4: Averages ---
            segments_data = {seg: [] for seg in segment_fallbacks}
            for i in range(len(vertices)):
                nz = normals[i][2]
                color = vertex_colors[i]
                is_background = color[0] > 0.90 and color[1] > 0.90 and color[2] > 0.90
                if nz > 0.35 and not is_background:
                    seg = get_segment_name(vertices[i])
                    segments_data[seg].append(color[:3])

            average_colors = {}
            for seg in segment_fallbacks:
                if len(segments_data[seg]) > 10:
                    arr = np.array(segments_data[seg])
                    average_colors[seg] = np.append(np.mean(arr, axis=0), 1.0)
                else:
                    average_colors[seg] = segment_fallbacks[seg].copy()

            # --- Phase 5: Smooth boundaries ---
            TRANSITION_ZONE = 0.03

            def get_segment_color_with_smooth_boundary(v):
                y = v[1]
                seg = get_segment_name(v)
                base_color = average_colors[seg]

                boundary_checks = [
                    ("shoes_top", "shoes", "pants"),
                    ("pants_top", "pants", "tshirt"),
                    ("tshirt_top", "tshirt", "skin"),
                    ("skin_top", "skin", "hair"),
                ]
                for bname, seg_below, seg_above in boundary_checks:
                    by = boundary_y[bname]
                    if abs(y - by) < TRANSITION_ZONE:
                        t = (y - (by - TRANSITION_ZONE)) / (2 * TRANSITION_ZONE)
                        t = np.clip(t, 0.0, 1.0)
                        t = t * t * (3 - 2 * t)
                        c_below = average_colors[seg_below]
                        c_above = average_colors[seg_above]
                        blended = (1 - t) * c_below + t * c_above
                        blended[3] = 1.0
                        return blended
                return base_color

            # --- Phase 6: Apply textures ---
            new_colors = []
            rng = np.random.RandomState(42)

            for i in range(len(vertices)):
                v = vertices[i]
                n = normals[i]
                tex_color = vertex_colors[i]
                nz = n[2]

                seg_color = get_segment_color_with_smooth_boundary(v)

                noise_intensity = 0.025
                noise = rng.uniform(-noise_intensity, noise_intensity, 3)
                seg_color_varied = seg_color.copy()
                seg_color_varied[:3] = np.clip(seg_color[:3] + noise, 0.0, 1.0)
                seg_color_varied[3] = 1.0

                if nz >= 0.15:
                    blend = 0.0
                elif nz <= -0.10:
                    blend = 1.0
                else:
                    blend = (0.15 - nz) / 0.25

                final_col = [
                    (1.0 - blend) * tex_color[j] + blend * seg_color_varied[j] for j in range(4)
                ]
                final_col_uint8 = [int(np.clip(c * 255.0, 0, 255)) for c in final_col]
                new_colors.append(final_col_uint8)

            mesh.visual.vertex_colors = np.array(new_colors, dtype=np.uint8)
            
            # Export to GLB
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            glb_data = mesh.export(file_type="glb")
            with open(output_path, "wb") as f_out:
                f_out.write(glb_data)
                
            print(f">>> [SUCCESS] 4D-Humans model compiled successfully at: {output_path}")
            return True
            
        except Exception as e:
            print(f"[ERROR] 4D-Humans execution failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

hmr2_service = Hmr2Service()
