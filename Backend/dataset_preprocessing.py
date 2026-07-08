import os
import sys
import json
import uuid
import argparse
import time
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# Resolve paths relative to this script's directory (Backend/)
backend_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(backend_dir, ".."))

# Add Backend folder to sys.path to support importing local services
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

# Global default dataset directory paths relative to the project root
DEFAULT_RAW_DIR = os.path.join(project_root, "data", "dataset_myntra")
DEFAULT_OUT_DIR = os.path.join(project_root, "data", "dataset_processed")

try:
    from services.pose_service import pose_service
    from services.segmentation_service import segmentation_service
except ImportError as e:
    print(f"[ERROR] Failed to import pose/segmentation services: {str(e)}")
    print("Please ensure this script resides inside the Backend/ folder.")


SUPPORTED_MIME_EXTS = (".jpg", ".jpeg", ".png")

def get_torso_polygon(landmarks, h, w):
    """
    Extracts the bounding polygon coordinates for the upper torso based on 33 landmarks.
    """
    if len(landmarks) <= 24:
        return None
        
    l_shoulder = landmarks[11]
    r_shoulder = landmarks[12]
    l_hip = landmarks[23]
    r_hip = landmarks[24]
    
    if l_shoulder["visibility"] < 0.4 or r_shoulder["visibility"] < 0.4:
        return None
        
    ls_x, ls_y = int(l_shoulder["x"] * w), int(l_shoulder["y"] * h)
    rs_x, rs_y = int(r_shoulder["x"] * w), int(r_shoulder["y"] * h)
    lh_x, lh_y = int(l_hip["x"] * w), int(l_hip["y"] * h)
    rh_x, rh_y = int(r_hip["x"] * w), int(r_hip["y"] * h)
    
    torso_points = [
        [rs_x, rs_y],
        [ls_x, ls_y],
        [lh_x, lh_y],
        [rh_x, rh_y]
    ]
    return np.array(torso_points, dtype=np.int32)

def extract_clothing_fallback(image_path, landmarks, output_path):
    """
    Pose-guided and HSV color-thresholded fallback cropping.
    Excludes model skin and background, leaving clean transparent garments.
    """
    img = cv2.imread(image_path)
    if img is None:
        return False
        
    h, w, c = img.shape
    torso_poly = get_torso_polygon(landmarks, h, w)
    
    if torso_poly is None:
        ymin, ymax = int(h * 0.18), int(h * 0.75)
        xmin, xmax = int(w * 0.15), int(w * 0.85)
    else:
        xmin = int(np.min(torso_poly[:, 0]))
        xmax = int(np.max(torso_poly[:, 0]))
        ymin = int(np.min(torso_poly[:, 1]))
        ymax = int(np.max(torso_poly[:, 1]))
        
        x_pad = int((xmax - xmin) * 0.35)
        y_pad_top = int((ymax - ymin) * 0.18)
        y_pad_bot = int((ymax - ymin) * 0.22)
        
        xmin = max(0, xmin - x_pad)
        xmax = min(w, xmax + x_pad)
        ymin = max(0, ymin - y_pad_top)
        ymax = min(h, ymax + y_pad_bot)
        
    cropped_rgb = img[ymin:ymax, xmin:xmax]
    ch, cw, cc = cropped_rgb.shape
    if ch == 0 or cw == 0:
        return False
        
    gray = cv2.cvtColor(cropped_rgb, cv2.COLOR_BGR2GRAY)
    _, bg_mask = cv2.threshold(gray, 246, 255, cv2.THRESH_BINARY_INV)
    
    hsv = cv2.cvtColor(cropped_rgb, cv2.COLOR_BGR2HSV)
    lower_skin1 = np.array([0, 15, 60], dtype=np.uint8)
    upper_skin1 = np.array([22, 160, 255], dtype=np.uint8)
    skin_mask1 = cv2.inRange(hsv, lower_skin1, upper_skin1)
    
    lower_skin2 = np.array([165, 15, 60], dtype=np.uint8)
    upper_skin2 = np.array([180, 160, 255], dtype=np.uint8)
    skin_mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
    
    full_skin_mask = cv2.bitwise_or(skin_mask1, skin_mask2)
    clothing_mask = cv2.bitwise_and(bg_mask, cv2.bitwise_not(full_skin_mask))
    
    torso_mask = np.zeros((ch, cw), dtype=np.uint8)
    
    if len(landmarks) > 24:
        l_shoulder = landmarks[11]
        r_shoulder = landmarks[12]
        l_hip = landmarks[23]
        r_hip = landmarks[24]
        
        local_collar_y = int(((l_shoulder["y"] + r_shoulder["y"]) / 2) * h) - ymin
        
        local_poly = []
        for idx in [11, 13, 15, 23, 24, 16, 14, 12]:
            if idx < len(landmarks):
                lx = int(landmarks[idx]["x"] * w) - xmin
                ly = int(landmarks[idx]["y"] * h) - ymin
                lx = max(0, min(cw - 1, lx))
                ly = max(0, min(ch - 1, ly))
                local_poly.append([lx, ly])
                
        if len(local_poly) > 2:
            pts = np.array(local_poly, dtype=np.int32)
            cv2.fillPoly(torso_mask, [pts], 255)
            torso_mask = cv2.dilate(torso_mask, np.ones((7, 7), np.uint8), iterations=3)
            clothing_mask = cv2.bitwise_and(clothing_mask, torso_mask)
            cv2.rectangle(clothing_mask, (0, 0), (cw, max(0, local_collar_y - int(ch * 0.05))), 0, -1)
    else:
        cv2.ellipse(clothing_mask, (cw // 2, ch // 2), (int(cw * 0.45), int(ch * 0.48)), 0, 255, -1)
        
    clothing_mask = cv2.medianBlur(clothing_mask, 5)
    clothing_mask = cv2.GaussianBlur(clothing_mask, (9, 9), 0)
    
    pil_img = Image.fromarray(cv2.cvtColor(cropped_rgb, cv2.COLOR_BGR2RGB))
    feathered_mask = Image.fromarray(clothing_mask).filter(ImageFilter.GaussianBlur(radius=3))
    
    rgba_img = pil_img.convert("RGBA")
    rgba_img.putalpha(feathered_mask)
    
    rgba_img.save(output_path, "PNG")
    return True

# Global cache for Segformer
_segformer_processor = None
_segformer_model = None

def get_segformer_instance():
    global _segformer_processor, _segformer_model
    if _segformer_model is not None:
        return _segformer_processor, _segformer_model
        
    print(">>> Initializing lightweight dynamic Fashion-Parser (SegFormer)...")
    from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
    import torch
    
    model_name = "mattmdjaga/segformer_b2_clothes"
    _segformer_processor = AutoImageProcessor.from_pretrained(model_name)
    _segformer_model = AutoModelForSemanticSegmentation.from_pretrained(model_name)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Segformer mapped to acceleration hardware: [{device.upper()}] ---")
    _segformer_model = _segformer_model.to(device)
    _segformer_model.eval()
    
    return _segformer_processor, _segformer_model

def extract_clothing_dynamic(image_path, output_path):
    """
    Fashion Parser utilizing pre-trained SegFormer Clothes model.
    Isolates garments and strips background, skin, and face.
    """
    import torch
    import torch.nn as nn
    
    pil_img = Image.open(image_path).convert("RGB")
    w, h = pil_img.size
    
    try:
        processor, model = get_segformer_instance()
    except Exception as e:
        print(f"   Segformer initialization failed: {str(e)}")
        return False
    
    inputs = processor(images=pil_img, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits.cpu()
        
    upsampled_logits = nn.functional.interpolate(
        logits,
        size=(h, w),
        mode="bilinear",
        align_corners=False,
    )
    
    pred_seg = upsampled_logits.argmax(dim=1)[0].numpy()
    
    # 4: Upper-clothes, 5: Skirt, 6: Pants (optional fallback), 7: Dress
    target_classes = [4, 5, 7] 
    clothing_mask_bool = np.isin(pred_seg, target_classes)
    
    if not np.any(clothing_mask_bool):
        clothing_mask_bool = np.isin(pred_seg, [4, 5, 6, 7])
        
    if not np.any(clothing_mask_bool):
        return False
        
    binary_mask = np.zeros((h, w), dtype=np.uint8)
    binary_mask[clothing_mask_bool] = 255
    
    kernel = np.ones((5, 5), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    
    num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(binary_mask)
    if num_labels > 1:
        sizes = stats[1:, cv2.CC_STAT_AREA]
        max_size = np.max(sizes) if len(sizes) > 0 else 0
        
        clean_mask = np.zeros_like(binary_mask)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= 0.15 * max_size:
                clean_mask[labels_im == i] = 255
        binary_mask = clean_mask
    
    y_indices, x_indices = np.where(binary_mask)
    if len(x_indices) == 0 or len(y_indices) == 0:
        return False
        
    ymin, ymax = np.min(y_indices), np.max(y_indices)
    xmin, xmax = np.min(x_indices), np.max(x_indices)
    
    padding = 15
    ymin = max(0, ymin - padding)
    ymax = min(h, ymax + padding)
    xmin = max(0, xmin - padding)
    xmax = min(w, xmax + padding)
    
    feathered_mask = cv2.GaussianBlur(binary_mask, (7, 7), 0)
    
    img_bgr = cv2.imread(image_path)
    cropped_bgr = img_bgr[ymin:ymax, xmin:xmax]
    cropped_mask = feathered_mask[ymin:ymax, xmin:xmax]
    
    rgba = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = cropped_mask
    
    cv2.imwrite(output_path, rgba)
    return True

def auto_discover_dataset(search_root):
    """
    Tries to find dataset_myntra directory by scanning root and subfolders dynamically.
    """
    candidates = [
        os.path.join(search_root, "VirtualTryOn", "app", "src", "assets", "dataset_myntra"),
        os.path.join(search_root, "3DModel", "Android", "app", "src", "assets", "dataset_myntra"),
        os.path.join(search_root, "dataset_myntra")
    ]
    
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
            
    # Proactive walk search
    for root, dirs, files in os.walk(search_root):
        if "dataset_myntra" in dirs:
            return os.path.join(root, "dataset_myntra")
            
    return None

def main():
    parser = argparse.ArgumentParser(description="Master Preprocessing Script - Isolate Garments for 16GB Dataset")
    parser.add_argument("--limit", type=int, default=0, help="Max items to process (0 = process the entire 16GB dataset)")
    parser.add_argument("--raw-dir", type=str, default="", help="Path to raw dataset folder (leave empty for default)")
    parser.add_argument("--out-dir", type=str, default="", help="Path to save output (leave empty for default)")
    parser.add_argument("--dry-run", action="store_true", help="Perform import/path checks and exit immediately without processing")
    args = parser.parse_args()
    
    # 1. Dry Run Validation Check (For 4GB VRAM or local validation testing)
    if args.dry_run:
        print("\n================== DATASET PREPROCESSING DRY RUN ==================")
        print("[CHECK 1/4] Checking python dependencies...")
        try:
            import cv2
            import numpy as np
            import PIL
            import torch
            import transformers
            print("   -> Core dependencies are present.")
        except ImportError as e:
            print(f"   [ERROR] Missing dependency: {str(e)}")
            sys.exit(1)
            
        print("[CHECK 2/4] Checking local project services...")
        try:
            from services.pose_service import pose_service
            from services.segmentation_service import segmentation_service
            print("   -> Pose and Segmentation services imported successfully.")
        except ImportError as e:
            print(f"   [ERROR] Service import failed: {str(e)}")
            sys.exit(1)
            
        print("[CHECK 3/4] Checking hardware acceleration (CUDA/VRAM)...")
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"   -> CUDA is AVAILABLE on: [{device_name}]")
            print(f"   -> Total VRAM detected: {total_vram_gb:.2f} GB")
            if total_vram_gb < 6.0:
                print("   [NOTE] Your laptop has a 4GB VRAM GPU. This is perfect for this dry-run / proxy setup!")
                print("          Since your laptop handles the frontend/backend proxy, the heavy AI try-on models")
                print("          will run on the remote PC workstation (16GB VRAM) to prevent local GPU crashes.")
        else:
            print("   -> CUDA is NOT available (CPU Mode).")
            
        print("[CHECK 4/4] Checking dataset folders...")
        raw_root = args.raw_dir or DEFAULT_RAW_DIR
        out_dir = args.out_dir or DEFAULT_OUT_DIR
        print(f"   -> Expected Raw Dataset: {raw_root}")
        print(f"   -> Expected Output Folder: {out_dir}")
        
        if os.path.exists(raw_root):
            print("   -> Raw dataset directory EXISTS.")
        else:
            print("   [WARNING] Raw dataset directory not found at the default path.")
            print("             (This is completely fine if you've already preprocessed the data and saved it.)")
            
        if os.path.exists(out_dir):
            print("   -> Processed output directory EXISTS.")
            catalog_file = os.path.join(out_dir, "catalog.json")
            if os.path.exists(catalog_file):
                print(f"   -> Catalog database 'catalog.json' EXISTS inside output folder.")
        
        print("\n[SUCCESS] Dry run check completed successfully! Script compiled and imports are 100% correct.")
        sys.exit(0)

    # 2. Resolve Raw Directory
    raw_root = args.raw_dir
    if not raw_root:
        raw_root = DEFAULT_RAW_DIR
            
    if not os.path.exists(raw_root):
        print(f"\n[CRITICAL ERROR] Raw dataset directory does not exist: {raw_root}")
        print("Please place your 'dataset_myntra' folder in the workspace root or pass its location using --raw-dir.")
        sys.exit(1)
        
    # 3. Resolve Output Directory
    out_dir = args.out_dir or DEFAULT_OUT_DIR

        
    os.makedirs(out_dir, exist_ok=True)
    print(f"[OUTPUT] Isolated transparent garments will be saved to: {out_dir}")
    
    # 3. Discover Images Grouped by Subdirectory
    print("\nScanning dataset directories (this may take a minute for 16GB)...")
    scan_configs = [
        {"gender": "men", "path": os.path.join(raw_root, "Final_Myntra_MEN")},
        {"gender": "women", "path": os.path.join(raw_root, "Final_Myntra_WOMEN")}
    ]
    
    men_images = []
    women_images = []
    
    for config in scan_configs:
        gender = config["gender"]
        folder_path = config["path"]
        if not os.path.exists(folder_path):
            print(f"   [SKIP] Folder not found: {folder_path}")
            continue
            
        for root, dirs, files in os.walk(folder_path):
            # Sort subdirectories and files to ensure stable, alphabetical walk order
            dirs.sort()
            files.sort()
            rel_dir = os.path.relpath(root, raw_root)
            for file in files:
                if file.lower().endswith(SUPPORTED_MIME_EXTS):
                    img_item = {
                        "gender": gender,
                        "file_path": os.path.join(root, file),
                        "filename": file,
                        "rel_dir": rel_dir
                    }
                    if gender == "men":
                        men_images.append(img_item)
                    else:
                        women_images.append(img_item)
                        
    total_images = len(men_images) + len(women_images)
    print(f"Finished scanning. Discovered {len(men_images)} Men and {len(women_images)} Women source images (Total: {total_images}).")
    if total_images == 0:
        print("[ABORT] No source images discovered inside 'Final_Myntra_MEN' or 'Final_Myntra_WOMEN'.")
        sys.exit(1)
        
    # 4. Set Limits globally across the deterministic order
    limit = args.limit
    if limit > 0:
        print(f"[LIMIT] Cap set to process first {limit} images across all subfolders.")
        combined = men_images + women_images
        combined = combined[:limit]
        
        # Split back to lists capped by limit
        men_images = [img for img in combined if img["gender"] == "men"]
        women_images = [img for img in combined if img["gender"] == "women"]
        total_to_process = len(combined)
    else:
        print("[LIMIT] Process-all active. Scanning all 16GB of images!")
        total_to_process = total_images
        
    # 5. Helper to chunk image lists into ~1GB segments based on file sizes
    def chunk_images_by_size(images, chunk_size_bytes=1024 * 1024 * 1024):
        chunks = []
        current_chunk = []
        current_chunk_size = 0
        
        for img in images:
            try:
                sz = os.path.getsize(img["file_path"])
            except Exception:
                sz = 150 * 1024  # Default to 150KB if size lookup fails
                
            current_chunk.append(img)
            current_chunk_size += sz
            
            if current_chunk_size >= chunk_size_bytes:
                chunks.append({
                    "images": current_chunk,
                    "size_bytes": current_chunk_size
                })
                current_chunk = []
                current_chunk_size = 0
                
        if current_chunk:
            chunks.append({
                "images": current_chunk,
                "size_bytes": current_chunk_size
            })
        return chunks

    men_chunks = chunk_images_by_size(men_images)
    women_chunks = chunk_images_by_size(women_images)
    
    # Generate mock metadata helper names
    men_names = ["Classic Slim Fit Shirt", "Casual Cotton Tee", "Premium Summer T-Shirt", "Urban Polo Collar Shirt", "Breathable Athletic Top", "Streetwear Oversized Tee"]
    women_names = ["Chic Floral Dress", "Elegant Flutter-Sleeve Top", "Casual Knit T-Shirt", "Sleek Cropped Tank Top", "V-Neck Designer Blouse", "Comfortable Oversized Kurta"]
    brands = ["Myntra Select", "Roadster", "HRX by Hrithik Roshan", "WROGN", "Mast & Harbour", "Anouk"]
    
    catalog_items = []
    total_processed_count = 0
    processed_via_segformer = 0
    processed_via_fallback = 0
    skipped_no_pose = 0
    skipped_errors = 0
    
    start_time = time.time()
    
    print("\n================== STARTING PREPROCESSING BATCH ==================")
    
    gender_groups = [
        {"label": "Men", "chunks": men_chunks},
        {"label": "Women", "chunks": women_chunks}
    ]
    
    for group in gender_groups:
        gender_label = group["label"]
        chunks = group["chunks"]
        num_chunks = len(chunks)
        
        if num_chunks == 0:
            continue
            
        for chunk_idx, chunk in enumerate(chunks):
            chunk_images = chunk["images"]
            chunk_size_bytes = chunk["size_bytes"]
            chunk_size_mb = chunk_size_bytes / (1024 * 1024)
            chunk_total = len(chunk_images)
            
            chunk_start_time = time.time()
            chunk_processed_count = 0
            chunk_skipped_count = 0
            
            prefix = f"{gender_label} Chunk {chunk_idx + 1}/{num_chunks} ({chunk_size_mb:.1f} MB)"
            
            for idx, item in enumerate(chunk_images):
                gender = item["gender"]
                img_path = item["file_path"]
                fname = item["filename"]
                rel_dir = item["rel_dir"]
                
                # Dynamic in-place progress update
                elapsed = time.time() - chunk_start_time
                avg_speed = chunk_processed_count / elapsed if elapsed > 0 else 0
                remaining = chunk_total - (idx + 1)
                eta_seconds = remaining / avg_speed if avg_speed > 0 else 0
                
                m, s = divmod(int(eta_seconds), 60)
                h, m = divmod(m, 60)
                eta_str = f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"
                
                percent = ((idx + 1) / chunk_total) * 100
                bar_length = 20
                filled_length = int(bar_length * (idx + 1) // chunk_total)
                bar = "=" * filled_length + "-" * (bar_length - filled_length)
                
                sys.stdout.write(
                    f"\r[{prefix}] |{bar}| {percent:.1f}% ({idx+1}/{chunk_total}) | Speed: {avg_speed:.2f} img/s | ETA: {eta_str}"
                )
                sys.stdout.flush()
                
                # 1. Pose landmarks check
                try:
                    landmarks = pose_service.extract_landmarks(img_path)
                except Exception:
                    skipped_errors += 1
                    chunk_skipped_count += 1
                    continue
                    
                if not landmarks:
                    skipped_no_pose += 1
                    chunk_skipped_count += 1
                    continue
                    
                # 2. Extract clothing into the preserved output subdirectory matching source
                target_out_dir = os.path.join(out_dir, rel_dir)
                os.makedirs(target_out_dir, exist_ok=True)
                
                # Keep exact same file name, change extension to .png
                base_name = os.path.splitext(fname)[0]
                out_filename = f"{base_name}.png"
                output_file_path = os.path.join(target_out_dir, out_filename)
                
                success = False
                used_segformer = False
                used_fallback = False
                try:
                    success = extract_clothing_dynamic(img_path, output_file_path)
                    if success:
                        used_segformer = True
                    else:
                        success = extract_clothing_fallback(img_path, landmarks, output_file_path)
                        if success:
                            used_fallback = True
                except Exception:
                    try:
                        success = extract_clothing_fallback(img_path, landmarks, output_file_path)
                        if success:
                            used_fallback = True
                    except Exception:
                        skipped_errors += 1
                        chunk_skipped_count += 1
                        success = False
                        
                if not success:
                    if os.path.exists(output_file_path):
                        try:
                            os.remove(output_file_path)
                        except Exception:
                            pass
                    chunk_skipped_count += 1
                    continue
                    
                # 3. Create metadata entry with preserved relative subdirectory inside imageUrl
                brand = brands[total_processed_count % len(brands)]
                category = "T-Shirt" if "Tee" in fname or "Tshirt" in fname or total_processed_count % 3 == 0 else ("Shirt" if total_processed_count % 3 == 1 else "Top")
                if gender == "women" and total_processed_count % 2 == 1:
                    category = "Dress"
                    
                if gender == "men":
                    name = f"{brand} Men's {men_names[total_processed_count % len(men_names)]}"
                else:
                    name = f"{brand} Women's {women_names[total_processed_count % len(women_names)]}"
                    
                rel_url_path = os.path.join(rel_dir, out_filename).replace("\\", "/")
                clothing_id = f"myntra_{gender}_{total_processed_count + 1}"
                
                catalog_items.append({
                    "id": clothing_id,
                    "name": name,
                    "category": category,
                    "imageUrl": f"/static/clothing/{rel_url_path}",
                    "sizes": ["S", "M", "L", "XL"],
                    "brand": brand,
                    "gender": gender
                })
                
                if used_segformer:
                    processed_via_segformer += 1
                elif used_fallback:
                    processed_via_fallback += 1
                    
                chunk_processed_count += 1
                total_processed_count += 1
                
            # Chunk completed: print final clean completion line
            elapsed_chunk = time.time() - chunk_start_time
            final_speed = chunk_processed_count / elapsed_chunk if elapsed_chunk > 0 else 0
            sys.stdout.write(
                f"\r[OK] [{prefix}] Completed! |{'='*20}| 100.0% (Processed: {chunk_processed_count} | Skipped: {chunk_skipped_count}) | Speed: {final_speed:.2f} img/s\n"
            )
            sys.stdout.flush()
            
    # 6. Compile and Write Master Catalog JSON database
    catalog_json_path = os.path.join(out_dir, "catalog.json")
    try:
        with open(catalog_json_path, "w") as f:
            json.dump(catalog_items, f, indent=4)
        print("\n================== BATCH PROCESSING COMPLETE ==================")
        print(f"[OK] Successfully processed: {total_processed_count} clothing items (SegFormer: {processed_via_segformer} | Fallback: {processed_via_fallback})")
        print(f"[SKIP] Skipped (No pose detected): {skipped_no_pose}")
        print(f"[ERR] Skipped (Errors/Corrupted): {skipped_errors}")
        print(f"Total time elapsed: {(time.time() - start_time) / 60:.2f} minutes")
        print(f"Master Catalog Database compiled successfully at: {catalog_json_path}")
    except Exception as e:
        print(f"[ERROR] Failed to write master catalog.json: {str(e)}")

if __name__ == "__main__":
    main()
