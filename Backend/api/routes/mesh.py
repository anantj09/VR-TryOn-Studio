from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status, Header
from schemas.tryon import MeasurementsSchema
from core.config import settings
from services.pose_service import pose_service
from services.segmentation_service import segmentation_service
from core.database import user_profiles_db
from pydantic import BaseModel
from PIL import Image
from typing import Optional
import uuid
import os
import shutil
import numpy as np
import json
import httpx

router = APIRouter(tags=["mesh"])

SUPPORTED_MIME_TYPES = ["image/jpeg", "image/png", "image/jpg"]

class MeshResponse(BaseModel):
    meshUrl: str
    measurements: MeasurementsSchema
    tryonImageUrl: Optional[str] = None

class TryonResponse(BaseModel):
    tryonId: str
    tryonImageUrl: str

@router.post("/tryon-only", response_model=TryonResponse)
async def generate_tryon_only(
    photo: UploadFile = File(..., description="The user portrait"),
    clothing_id: str = Form(..., description="ID of the clothing item from the catalog"),
    use_super_resolution: bool = Form(True, description="Whether to apply AI super resolution and face restoration"),
    x_gpu_server_url: Optional[str] = Header(None, alias="X-GPU-Server-URL")
):
    if not x_gpu_server_url:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="GPU Server URL is not configured. Configure it in settings."
        )

    # 1. Validate photo MIME type
    if photo.content_type not in SUPPORTED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: {photo.content_type}. Only JPEG and PNG are allowed."
        )

    # 2. Save photo to temporary folder
    temp_dir = settings.temp_dir_path
    os.makedirs(temp_dir, exist_ok=True)
    job_id = str(uuid.uuid4())
    file_ext = os.path.splitext(photo.filename)[1] or ".jpg"
    person_img_path = os.path.join(temp_dir, f"{job_id}_input{file_ext}")

    try:
        with open(person_img_path, "wb") as buffer:
            shutil.copyfileobj(photo.file, buffer)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save temporary photo: {str(e)}"
        )
    finally:
        await photo.close()

    # 3. Locate clothing image path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    static_meshes_dir = os.path.abspath(os.path.join(current_dir, "..", "..", "..", "data", "meshes"))
    
    catalog_file = None
    for p in ["dataset_processed/catalog.json", "clothing/catalog.json"]:
        p_abs = os.path.abspath(os.path.join(static_meshes_dir, "..", p))
        if os.path.exists(p_abs):
            catalog_file = p_abs
            break

    if not catalog_file:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Catalog database not found."
        )

    with open(catalog_file, "r") as f:
        catalog_data = json.load(f)

    selected_item = next((item for item in catalog_data if item.get("id") == clothing_id), None)
    if not selected_item:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Clothing item {clothing_id} not found in catalog."
        )

    img_url = selected_item.get("imageUrl", "")
    relative_img_path = img_url.replace("/static/clothing/", "dataset_processed/")
    if relative_img_path.startswith("/static/"):
        relative_img_path = img_url.replace("/static/", "")

    clothing_img_path = os.path.abspath(os.path.join(static_meshes_dir, "..", relative_img_path))
    if not os.path.exists(clothing_img_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Clothing image file not found on disk."
        )

    # 4. Call GPU PC `/api/v1/tryon`
    gpu_endpoint = f"{x_gpu_server_url.rstrip('/')}/api/v1/tryon"
    files = {
        "person_image": (os.path.basename(person_img_path), open(person_img_path, "rb"), "image/png"),
        "garment_image": (os.path.basename(clothing_img_path), open(clothing_img_path, "rb"), "image/png")
    }
    data = {
        "cloth_type": "upper",
        "use_super_resolution": "true" if use_super_resolution else "false"
    }

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(gpu_endpoint, files=files, data=data)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to connect to GPU Server: {str(e)}"
        )
    finally:
        for f_handle in files.values():
            f_handle[1].close()
        # clean up input temp file
        if os.path.exists(person_img_path):
            os.remove(person_img_path)

    if response.status_code != 200:
        raise HTTPException(
            status_code=response.status_code,
            detail=f"GPU Server returned error: {response.text}"
        )

    # Save to data/tryon_outputs/
    tryon_dir = os.path.abspath(os.path.join(static_meshes_dir, "..", "tryon_outputs"))
    os.makedirs(tryon_dir, exist_ok=True)
    tryon_file_path = os.path.join(tryon_dir, f"{job_id}.png")

    try:
        with open(tryon_file_path, "wb") as f:
            f.write(response.content)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save try-on result: {str(e)}"
        )

    return TryonResponse(
        tryonId=job_id,
        tryonImageUrl=f"/static/tryon_outputs/{job_id}.png"
    )

def convert_splat_ply_to_colored_ply(ply_path: str):
    import struct
    
    if not os.path.exists(ply_path):
        print(f"[CONVERSION ERROR] File not found: {ply_path}")
        return
        
    try:
        with open(ply_path, "rb") as f:
            header = ""
            while True:
                line = f.readline().decode("utf-8", errors="ignore")
                header += line
                if "end_header" in line:
                    break
                    
            vertex_count = 0
            for line in header.split("\n"):
                if line.startswith("element vertex"):
                    vertex_count = int(line.split()[-1])
                    break
                    
            if vertex_count == 0:
                print("[CONVERSION] No vertices found in PLY header.")
                return
                
            binary_data = f.read()
            
        # Standard Splat PLY vertex size is 68 bytes (17 floats):
        # x, y, z, nx, ny, nz, f_dc_0, f_dc_1, f_dc_2, opacity, scale_0, scale_1, scale_2, rot_0, rot_1, rot_2, rot_3
        vertex_size = 68
        expected_size = vertex_count * vertex_size
        if len(binary_data) < expected_size:
            print(f"[CONVERSION ERROR] Binary data size ({len(binary_data)}) is less than expected ({expected_size}). Skipping conversion.")
            return

        new_header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {vertex_count}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property float nx\n"
            "property float ny\n"
            "property float nz\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            "end_header\n"
        )
        
        SH_C0 = 0.28209479177387814
        converted_vertices = []
        
        for i in range(vertex_count):
            offset = i * vertex_size
            vertex_floats = struct.unpack_from("<17f", binary_data, offset)
            
            x, y, z = vertex_floats[0], vertex_floats[1], vertex_floats[2]
            nx, ny, nz = vertex_floats[3], vertex_floats[4], vertex_floats[5]
            f_dc_0, f_dc_1, f_dc_2 = vertex_floats[6], vertex_floats[7], vertex_floats[8]
            
            # Convert Spherical Harmonics coefficients to RGB bytes [0, 255]
            r = int(max(0.0, min(1.0, 0.5 + SH_C0 * f_dc_0)) * 255)
            g = int(max(0.0, min(1.0, 0.5 + SH_C0 * f_dc_1)) * 255)
            b = int(max(0.0, min(1.0, 0.5 + SH_C0 * f_dc_2)) * 255)
            
            converted_vertices.append(struct.pack("<6f3B", x, y, z, nx, ny, nz, r, g, b))
            
        with open(ply_path, "wb") as f_out:
            f_out.write(new_header.encode("utf-8"))
            f_out.write(b"".join(converted_vertices))
            
        print(f"[CONVERSION SUCCESS] Converted splat PLY to standard colored PLY: {ply_path}")
    except Exception as e:
        print(f"[CONVERSION ERROR] Exception during PLY conversion: {str(e)}")

@router.post("/generate-mesh", response_model=MeshResponse)
async def generate_user_mesh(
    photo: Optional[UploadFile] = File(None, description="The user portrait to reconstruct 3D body from"),
    tryon_id: Optional[str] = Form(None, description="Optional ID of a previously generated try-on image"),
    user_id: Optional[str] = Form(None, description="Optional user ID for profiling"),
    clothing_id: Optional[str] = Form(None, description="Optional ID of the target clothing item from catalog"),
    method: Optional[str] = Form("4dhumans", description="3D generation method: '4dhumans' or 'lhmpp'"),
    x_colab_tunnel_url: Optional[str] = Header(None, alias="X-Colab-Tunnel-URL"),
    x_gpu_server_url: Optional[str] = Header(None, alias="X-GPU-Server-URL")
):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    static_meshes_dir = os.path.abspath(os.path.join(current_dir, "..", "..", "..", "data", "meshes"))
    os.makedirs(static_meshes_dir, exist_ok=True)

    # 3. Resolve source image path
    if tryon_id:
        tryon_dir = os.path.abspath(os.path.join(static_meshes_dir, "..", "tryon_outputs"))
        person_img_path = os.path.join(tryon_dir, f"{tryon_id}.png")
        if not os.path.exists(person_img_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Tryon image not found."
            )
        job_id = tryon_id
    else:
        if not photo:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Either photo or tryon_id must be provided."
            )
        # 1. Validate MIME type
        if photo.content_type not in SUPPORTED_MIME_TYPES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file type: {photo.content_type}. Only JPEG and PNG are allowed."
            )

        # 2. Check if the file is readable image
        try:
            img = Image.open(photo.file)
            img.verify()
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is corrupted or not a valid image format."
            )
        finally:
            await photo.seek(0)

        job_id = str(uuid.uuid4())
        file_ext = os.path.splitext(photo.filename)[1] or ".jpg"
        unique_filename = f"{job_id}_input{file_ext}"
        temp_path = settings.temp_dir_path
        os.makedirs(temp_path, exist_ok=True)
        person_img_path = os.path.join(temp_path, unique_filename)

        # Save user photo
        try:
            with open(person_img_path, "wb") as buffer:
                shutil.copyfileobj(photo.file, buffer)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to save temporary photo: {str(e)}"
            )
        finally:
            await photo.close()

    # 4. Remote GPU Try-on (CatVTON) if clothing_id and GPU server address are provided and not already tryon_id
    tryon_image_url = None
    if tryon_id:
        tryon_image_url = f"/static/tryon_outputs/{tryon_id}.png"

    if clothing_id and x_gpu_server_url and not tryon_id:
        print(f"Executing remote try-on for clothing_id {clothing_id} on GPU server {x_gpu_server_url}...")
        try:
            # Locate catalog.json to find clothing item image relative path
            catalog_file = None
            for p in ["dataset_processed/catalog.json", "clothing/catalog.json"]:
                p_abs = os.path.abspath(os.path.join(static_meshes_dir, "..", p))
                if os.path.exists(p_abs):
                    catalog_file = p_abs
                    break

            if catalog_file:
                with open(catalog_file, "r") as f:
                    catalog_data = json.load(f)

                selected_item = next((item for item in catalog_data if item.get("id") == clothing_id), None)
                if selected_item:
                    img_url = selected_item.get("imageUrl", "")
                    # Convert static URL to disk path
                    relative_img_path = img_url.replace("/static/clothing/", "dataset_processed/")
                    if relative_img_path.startswith("/static/"):
                        relative_img_path = img_url.replace("/static/", "")

                    clothing_img_path = os.path.abspath(os.path.join(static_meshes_dir, "..", relative_img_path))

                    if os.path.exists(clothing_img_path):
                        # Call remote GPU server /api/v1/tryon
                        gpu_endpoint = f"{x_gpu_server_url.rstrip('/')}/api/v1/tryon"
                        
                        # Prepare files
                        files = {
                            "person_image": (os.path.basename(person_img_path), open(person_img_path, "rb"), "image/png"),
                            "garment_image": (os.path.basename(clothing_img_path), open(clothing_img_path, "rb"), "image/png")
                        }
                        data = {
                            "cloth_type": "upper"
                        }
                        
                        async with httpx.AsyncClient(timeout=60.0) as client:
                            response = await client.post(gpu_endpoint, files=files, data=data)
                        
                        # Close file handles
                        for f_handle in files.values():
                            f_handle[1].close()

                        if response.status_code == 200:
                            # Save try-on output to results folder so it scavenges automatically
                            results_dir = os.path.abspath(os.path.join(static_meshes_dir, "..", "results"))
                            os.makedirs(results_dir, exist_ok=True)
                            tryon_filename = f"{job_id}_tryon.png"
                            tryon_filepath = os.path.join(results_dir, tryon_filename)
                            
                            with open(tryon_filepath, "wb") as f_out:
                                f_out.write(response.content)

                            # Set the try-on image as input for MediaPipe pose estimation & 3D mesh building
                            person_img_path = tryon_filepath
                            tryon_image_url = f"/static/results/{tryon_filename}"
                            print(f"Remote try-on succeeded. Output saved to: {tryon_filename}")
                        else:
                            print(f"[WARNING] Remote GPU try-on failed with status {response.status_code}: {response.text}")
                    else:
                        print(f"[WARNING] Clothing item image not found on disk: {clothing_img_path}")
                else:
                    print(f"[WARNING] Clothing item ID {clothing_id} not found in catalog")
            else:
                print("[WARNING] catalog.json not found on disk")
        except Exception as e:
            print(f"[WARNING] Remote try-on error: {str(e)}. Falling back to original portrait for 3D model.")

    # 5. Extract landmarks via pose_service
    try:
        landmarks = pose_service.extract_landmarks(person_img_path)
    except Exception as e:
        if os.path.exists(person_img_path):
            os.remove(person_img_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error running pose landmark extraction: {str(e)}"
        )

    if not landmarks:
        if os.path.exists(person_img_path):
            os.remove(person_img_path)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No human pose detected. Please upload a clear photo showing a person's full body."
        )

    # 6. Calculate body sizing dimensions using landmarks
    try:
        l_shoulder = landmarks[11]
        r_shoulder = landmarks[12]
        l_hip = landmarks[23]
        
        shoulder_dist = np.sqrt((l_shoulder["x"] - r_shoulder["x"])**2 + (l_shoulder["y"] - r_shoulder["y"])**2)
        torso_len = np.sqrt((l_shoulder["x"] - l_hip["x"])**2 + (l_shoulder["y"] - l_hip["y"])**2)
        
        height_est_cm = 172.5 + (torso_len - 0.45) * 60.0
        chest_est_cm = 88.0 + (shoulder_dist - 0.22) * 110.0
        waist_est_cm = 78.0 + (shoulder_dist - 0.22) * 95.0
        hip_est_cm = chest_est_cm + 3.0
        
        # Clamp bounds
        height_est_cm = max(140.0, min(210.0, height_est_cm))
        chest_est_cm = max(70.0, min(130.0, chest_est_cm))
        waist_est_cm = max(60.0, min(120.0, waist_est_cm))
        hip_est_cm = max(70.0, min(135.0, hip_est_cm))
        
        measurements_dict = {
            "chestCm": round(chest_est_cm, 1),
            "waistCm": round(waist_est_cm, 1),
            "hipCm": round(hip_est_cm, 1),
            "heightCm": round(height_est_cm, 1)
        }
        
        # 7. Generate 3D human mannequin / digital twin
        success = False
        
        if method == "lhmpp":
            if x_gpu_server_url:
                gpu_lhm_endpoint = f"{x_gpu_server_url.rstrip('/')}/api/v1/reconstruct-lhmpp"
                print(f"Forwarding LHM++ 3D generation request to GPU server: {gpu_lhm_endpoint}")
                
                mesh_filename = f"{job_id}_mesh.ply"
                mesh_filepath = os.path.join(static_meshes_dir, mesh_filename)
                
                try:
                    # 180-second timeout to allow LHM++ execution and transfer
                    async with httpx.AsyncClient(timeout=180.0) as client:
                        with open(person_img_path, "rb") as f_img:
                            files = {"image": (os.path.basename(person_img_path), f_img, "image/png")}
                            response = await client.post(gpu_lhm_endpoint, files=files)
                            
                    if response.status_code == 200:
                        with open(mesh_filepath, "wb") as f_out:
                            f_out.write(response.content)
                        
                        # Convert splat color attributes to standard RGB properties on-the-fly
                        try:
                            convert_splat_ply_to_colored_ply(mesh_filepath)
                        except Exception as e:
                            print(f"[WARNING] Failed to convert splat colors: {str(e)}")
                            
                        success = True
                        print(f"LHM++ GPU visual 3D generation succeeded. Saved as: {mesh_filename}")
                    else:
                        print(f"[WARNING] LHM++ GPU generation failed with status {response.status_code}: {response.text}")
                except Exception as e:
                    print(f"[WARNING] LHM++ GPU connection error: {str(e)}. Falling back to local mannequin.")
            else:
                print("[WARNING] LHM++ requested but no x_gpu_server_url provided. Falling back to local mannequin.")

        if not success and x_colab_tunnel_url:
            # Route visual generation to Colab via Ngrok tunnel
            colab_endpoint = f"{x_colab_tunnel_url.rstrip('/')}/api/v1/colab/generate"
            print(f"Forwarding visual 3D generation request to Colab: {colab_endpoint}")
            
            mesh_filename = f"{job_id}_mesh.glb"
            mesh_filepath = os.path.join(static_meshes_dir, mesh_filename)
            
            try:
                # 180-second timeout to allow TRELLIS time to perform diffusion and GLB assembly
                async with httpx.AsyncClient(timeout=180.0) as client:
                    with open(person_img_path, "rb") as f_img:
                        files = {"photo": (os.path.basename(person_img_path), f_img, "image/png")}
                        response = await client.post(colab_endpoint, files=files)
                        
                if response.status_code == 200:
                    with open(mesh_filepath, "wb") as f_out:
                        f_out.write(response.content)
                    success = True
                    print(f"Colab visual 3D generation succeeded. Saved as: {mesh_filename}")
                else:
                    print(f"[WARNING] Colab generation failed with status {response.status_code}: {response.text}")
                    x_colab_tunnel_url = None # Trigger local fallback
            except Exception as e:
                print(f"[WARNING] Colab connection error: {str(e)}. Falling back to local mannequin.")
                x_colab_tunnel_url = None # Trigger local fallback
                
        if not success and method == "4dhumans":
            if x_gpu_server_url:
                gpu_4d_endpoint = f"{x_gpu_server_url.rstrip('/')}/api/v1/reconstruct-4d"
                print(f"Forwarding 4D-Humans 3D generation request to GPU server: {gpu_4d_endpoint}")
                
                mesh_filename = f"{job_id}_mesh.glb"
                mesh_filepath = os.path.join(static_meshes_dir, mesh_filename)
                
                import json
                try:
                    # 180-second timeout to allow 4D-Humans execution and transfer
                    async with httpx.AsyncClient(timeout=180.0) as client:
                        with open(person_img_path, "rb") as f_img:
                            files = {"image": (os.path.basename(person_img_path), f_img, "image/png")}
                            data = {}
                            if landmarks:
                                data["landmarks"] = json.dumps(landmarks)
                            response = await client.post(gpu_4d_endpoint, files=files, data=data)
                            
                    if response.status_code == 200:
                        with open(mesh_filepath, "wb") as f_out:
                            f_out.write(response.content)
                        success = True
                        print(f"4D-Humans GPU visual 3D generation succeeded. Saved as: {mesh_filename}")
                    else:
                        print(f"[WARNING] 4D-Humans GPU generation failed with status {response.status_code}: {response.text}")
                except Exception as e:
                    print(f"[WARNING] 4D-Humans GPU connection error: {str(e)}")
            else:
                print("[WARNING] 4D-Humans GPU requested but no x_gpu_server_url provided.")

        if not success:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="The remote GPU compute host is offline or busy. 3D human twin reconstruction requires an active GPU connection."
            )
            
        # Cache profile details
        target_user_id = user_id or "guest_user"
        if target_user_id != "guest_user":
            overall_fit = "Regular Fit"
            fit_rec = "We recommend size M."
            if chest_est_cm > 102.0:
                overall_fit = "Relaxed Fit"
                fit_rec = "We recommend size L."
            elif chest_est_cm < 88.0:
                overall_fit = "Slim Fit"
                fit_rec = "We recommend size S."
                
            user_profiles_db[target_user_id] = {
                "measurements": measurements_dict,
                "fitAnalysis": {
                    "shoulder": "Comfortable drape.",
                    "chest": "Tailored fit.",
                    "waist": "Comfortable waist shape.",
                    "overall": overall_fit,
                    "recommendation": fit_rec
                }
            }
            
        # Clean up temp inputs
        if os.path.exists(person_img_path) and not person_img_path.startswith(static_meshes_dir) and "results" not in person_img_path:
            os.remove(person_img_path)
            
        return MeshResponse(
            meshUrl=f"/static/meshes/{mesh_filename}",
            measurements=MeasurementsSchema(**measurements_dict),
            tryonImageUrl=tryon_image_url
        )
        
    except HTTPException as he:
        if os.path.exists(person_img_path):
            os.remove(person_img_path)
        raise he
    except Exception as e:
        if os.path.exists(person_img_path):
            os.remove(person_img_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"3D mesh reconstruction pipeline failure: {str(e)}"
        )

@router.post("/generate-digital-twin", response_model=MeshResponse)
async def generate_digital_twin(
    file: UploadFile = File(..., description="The user portrait to reconstruct 3D body from"),
    user_id: Optional[str] = Form(None, description="Optional user ID for profiling")
):
    # 1. Validate MIME type and image readability
    if file.content_type not in SUPPORTED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: {file.content_type}. Only JPEG and PNG are allowed."
        )

    try:
        img_bytes = await file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise Exception("Invalid image data")
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is corrupted or not a valid image format."
        )
    finally:
        await file.seek(0)

    # 2. Create temp file path to execute downstream services
    job_id = str(uuid.uuid4())
    file_ext = os.path.splitext(file.filename)[1] or ".jpg"
    unique_filename = f"{job_id}_input{file_ext}"
    temp_path = settings.temp_dir_path
    os.makedirs(temp_path, exist_ok=True)
    person_img_path = os.path.join(temp_path, unique_filename)

    try:
        # Write bytes locally for pre-processing tools
        with open(person_img_path, "wb") as buffer:
            buffer.write(img_bytes)

        # 3. RUN EXISTING FRONTEND PRE-PROCESSING TOOLS
        # Extract 2D MediaPipe joints
        landmarks = pose_service.extract_landmarks(person_img_path)
        if not landmarks:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No human pose detected. Please upload a clear photo showing a person's full body."
            )

        # Extract SAM foreground binary mask
        mask_filename = segmentation_service.generate_mask(person_img_path, landmarks)
        mask_path = os.path.join(settings.temp_dir_path, mask_filename)

        # Calculate sizing dimensions
        l_shoulder = landmarks[11]
        r_shoulder = landmarks[12]
        l_hip = landmarks[23]
        
        shoulder_dist = np.sqrt((l_shoulder["x"] - r_shoulder["x"])**2 + (l_shoulder["y"] - r_shoulder["y"])**2)
        torso_len = np.sqrt((l_shoulder["x"] - l_hip["x"])**2 + (l_shoulder["y"] - l_hip["y"])**2)
        
        height_est_cm = 172.5 + (torso_len - 0.45) * 60.0
        chest_est_cm = 88.0 + (shoulder_dist - 0.22) * 110.0
        waist_est_cm = 78.0 + (shoulder_dist - 0.22) * 95.0
        hip_est_cm = chest_est_cm + 3.0
        
        height_est_cm = max(140.0, min(210.0, height_est_cm))
        chest_est_cm = max(70.0, min(130.0, chest_est_cm))
        waist_est_cm = max(60.0, min(120.0, waist_est_cm))
        hip_est_cm = max(70.0, min(135.0, hip_est_cm))
        
        measurements_dict = {
            "chestCm": round(chest_est_cm, 1),
            "waistCm": round(waist_est_cm, 1),
            "hipCm": round(hip_est_cm, 1),
            "heightCm": round(height_est_cm, 1)
        }

        # 5. PARSE MULTI-LAYER OUTPUT PARAMETERS (Momentum Human Rig)
        # Compile GLB mesh
        current_dir = os.path.dirname(os.path.abspath(__file__))
        static_meshes_dir = os.path.abspath(os.path.join(current_dir, "..", "..", "..", "data", "meshes"))
        mesh_filename = f"{job_id}_mesh.gltf"
        mesh_filepath = os.path.join(static_meshes_dir, mesh_filename)

        success = mesh_service.generate_proportional_mannequin(
            chest_cm=chest_est_cm,
            waist_cm=waist_est_cm,
            hip_cm=hip_est_cm,
            height_cm=height_est_cm,
            output_path=mesh_filepath,
            person_img_path=person_img_path,
            mask_path=mask_path,
            landmarks=landmarks
        )

        # Cleanup temporary mask file
        try:
            if os.path.exists(mask_path):
                os.remove(mask_path)
        except Exception:
            pass

        if not success:
            raise Exception("MeshService failed to write GLTF output.")

        # Cache profile details
        target_user_id = user_id or "guest_user"
        if target_user_id != "guest_user":
            overall_fit = "Regular Fit"
            fit_rec = "We recommend size M."
            if chest_est_cm > 102.0:
                overall_fit = "Relaxed Fit"
                fit_rec = "We recommend size L."
            elif chest_est_cm < 88.0:
                overall_fit = "Slim Fit"
                fit_rec = "We recommend size S."
                
            user_profiles_db[target_user_id] = {
                "measurements": measurements_dict,
                "fitAnalysis": {
                    "shoulder": "Comfortable drape.",
                    "chest": "Tailored fit.",
                    "waist": "Comfortable waist shape.",
                    "overall": overall_fit,
                    "recommendation": fit_rec
                }
            }

        # Clean up temp inputs
        if os.path.exists(person_img_path):
            os.remove(person_img_path)

        return MeshResponse(
            meshUrl=f"/static/meshes/{mesh_filename}",
            measurements=MeasurementsSchema(**measurements_dict)
        )

    except Exception as e:
        if os.path.exists(person_img_path):
            os.remove(person_img_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"3D mesh reconstruction pipeline failure: {str(e)}"
        )


@router.get("/premade/list")
async def list_premade_assets():
    """
    Scans the data/premade/ directory for GLTF/GLB models.
    Merges custom display names and categories if defined in premade_mapping.json.
    """
    import os
    import json
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    premade_dir = os.path.abspath(os.path.join(current_dir, "..", "..", "..", "data", "premade"))
    
    if not os.path.exists(premade_dir):
        return []
        
    # Check for mapping overrides file
    mapping_path = os.path.join(premade_dir, "premade_mapping.json")
    mapping = {}
    if os.path.exists(mapping_path):
        try:
            with open(mapping_path, "r", encoding="utf-8") as f:
                mapping = json.load(f)
        except Exception as e:
            print(f"[WARNING] Failed to load premade mapping JSON: {str(e)}")
            
    assets = []
    try:
        for filename in os.listdir(premade_dir):
            if filename == "premade_mapping.json" or filename.startswith("."):
                continue
                
            file_path = os.path.join(premade_dir, filename)
            if os.path.isfile(file_path) and filename.lower().endswith((".glb", ".gltf")):
                size_bytes = os.path.getsize(file_path)
                
                # Check for custom override details
                override = mapping.get(filename, {})
                assets.append({
                    "filename": filename,
                    "sizeBytes": size_bytes,
                    "customName": override.get("name"),
                    "customCategory": override.get("category"),
                    "customScale": override.get("scale")
                })
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to scan premade assets directory: {str(e)}"
        )
        
    return assets


@router.get("/tryon/list")
async def list_tryon_images():
    """
    Lists previously generated try-on images from data/tryon_outputs/.
    """
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    tryon_dir = os.path.abspath(os.path.join(current_dir, "..", "..", "..", "data", "tryon_outputs"))
    
    if not os.path.exists(tryon_dir):
        return []
        
    tryons = []
    try:
        files = [f for f in os.listdir(tryon_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        files.sort(key=lambda x: os.path.getmtime(os.path.join(tryon_dir, x)), reverse=True)
        for filename in files:
            tryons.append({
                "tryonId": os.path.splitext(filename)[0],
                "imageUrl": f"/static/tryon_outputs/{filename}"
            })
    except Exception as e:
        print(f"[WARNING] Failed to list tryon images: {str(e)}")
        return []
    return tryons


@router.get("/meshes/list")
async def list_user_meshes():
    """
    Lists previously generated user meshes from data/meshes/.
    """
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    meshes_dir = os.path.abspath(os.path.join(current_dir, "..", "..", "..", "data", "meshes"))
    
    if not os.path.exists(meshes_dir):
        return []
        
    meshes = []
    try:
        files = [f for f in os.listdir(meshes_dir) if f.lower().endswith((".glb", ".gltf", ".ply", ".ksplat"))]
        files.sort(key=lambda x: os.path.getmtime(os.path.join(meshes_dir, x)), reverse=True)
        for filename in files:
            meshes.append({
                "filename": filename,
                "meshUrl": f"/static/meshes/{filename}"
            })
    except Exception as e:
        print(f"[WARNING] Failed to list user meshes: {str(e)}")
        return []
    return meshes


