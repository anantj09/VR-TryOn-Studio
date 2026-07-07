import os
import uuid
import shutil
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from catvton_service import catvton_service

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-initialize/download/load CatVTON weights on VRAM
    print(">>> Warm-starting GPU compute server (loading CatVTON model)...")
    try:
        catvton_service.initialize()
    except Exception as e:
        print(f"[STARTUP ERROR] Failed to initialize CatVTON: {str(e)}")
    yield

app = FastAPI(title="GPU Compute Server (CatVTON)", lifespan=lifespan)

TEMP_DIR = "/tmp/jobs"
os.makedirs(TEMP_DIR, exist_ok=True)

def cleanup_job_directory(job_dir: str):
    """
    Background worker task to delete temporary upload directories.
    """
    try:
        if os.path.exists(job_dir):
            shutil.rmtree(job_dir, ignore_errors=True)
            print(f">>> [CLEANUP] Deleted temporary folder: {job_dir}")
    except Exception as e:
        print(f"[CLEANUP ERROR] Failed to delete temporary folder {job_dir}: {str(e)}")

@app.post("/api/v1/tryon")
async def generate_tryon_only(
    background_tasks: BackgroundTasks,
    person_image: UploadFile = File(...),
    garment_image: UploadFile = File(...),
    cloth_type: str = Form("upper"),
    use_super_resolution: str = Form("true")
):
    """
    Runs ONLY the virtual try-on (CatVTON) on the remote GPU and returns the resulting PNG image.
    """
    job_id = str(uuid.uuid4())
    job_dir = os.path.join(TEMP_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)
    
    person_path = os.path.join(job_dir, "image.png")
    garment_path = os.path.join(job_dir, "garment.png")
    
    try:
        # Save person photo
        with open(person_path, "wb") as f:
            shutil.copyfileobj(person_image.file, f)
            
        # Save garment photo
        with open(garment_path, "wb") as f:
            shutil.copyfileobj(garment_image.file, f)
            
        print(f">>> Running virtual try-on (CatVTON-only) for job: {job_id}")
        tryon_img = catvton_service.run_tryon(person_path, garment_path, cloth_type=cloth_type)
        
        # Apply Super-Resolution & Face Restoration (if enabled)
        if use_super_resolution.lower() == "true":
            print(">>> Applying AI Super-Resolution (Real-ESRGAN + GFPGAN)...")
            from upscaler_service import upscaler_service
            tryon_img = upscaler_service.enhance_image(tryon_img)
        else:
            print(">>> Skipping AI Super-Resolution as requested.")
        
        # Save try-on output into job folder
        tryon_path = os.path.join(job_dir, "tryon_result.png")
        tryon_img.save(tryon_path)
        
        # Unload try-on, SR, and 4D-Humans models from GPU VRAM to leave it clean for LHM++
        print(">>> [VRAM] Reclaiming GPU memory post-generation...")
        try:
            catvton_service.unload_models()
        except Exception as err:
            print(f"[VRAM ERROR] Failed to unload CatVTON: {str(err)}")
            
        try:
            from upscaler_service import upscaler_service
            upscaler_service.unload_models()
        except Exception as err:
            print(f"[VRAM ERROR] Failed to unload upscaler: {str(err)}")
            
        try:
            from hmr2_service import hmr2_service
            hmr2_service.unload_models()
        except Exception as err:
            print(f"[VRAM ERROR] Failed to unload 4D-Humans: {str(err)}")
        
        if not os.path.exists(tryon_path):
            raise HTTPException(status_code=500, detail="CatVTON try-on image was not generated.")
            
        # Register background cleanup task
        background_tasks.add_task(cleanup_job_directory, job_dir)
        
        # Return generated PNG
        return FileResponse(tryon_path, media_type="image/png", filename=f"tryon_{job_id[:8]}.png")
        
    except Exception as e:
        # Cleanup immediately on failure
        shutil.rmtree(job_dir, ignore_errors=True)
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/reconstruct-lhmpp")
async def reconstruct_lhmpp(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...)
):
    """
    Runs LHM++ 3D Gaussian Splatting reconstruction on the uploaded image and returns the PLY file.
    """
    job_id = str(uuid.uuid4())
    job_dir = os.path.join(TEMP_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)
    
    # Save the image inside a subfolder named job_id so that LHMPP outputs under outputs/tpose_output/{job_id}.ply
    image_path = os.path.join(job_dir, "image.png")
    
    try:
        # Pre-emptive VRAM cleanup to prevent CUDA Out Of Memory crashes
        print(">>> [VRAM] Reclaiming GPU memory before LHM++ reconstruction...")
        try:
            catvton_service.unload_models()
        except Exception:
            pass
        try:
            from upscaler_service import upscaler_service
            upscaler_service.unload_models()
        except Exception:
            pass
        try:
            from hmr2_service import hmr2_service
            hmr2_service.unload_models()
        except Exception:
            pass

        with open(image_path, "wb") as f:
            shutil.copyfileobj(image.file, f)
            
        print(f">>> Running LHM++ reconstruction on GPU for job: {job_id}")
        
        # Import dynamically so server doesn't crash on startup if LHM++ is not installed
        from lhmpp_service import lhmpp_service
        ply_path = lhmpp_service.reconstruct_3d(job_id, image_path)
        
        if not os.path.exists(ply_path):
            raise HTTPException(status_code=500, detail="LHM++ PLY file was not generated.")
            
        # Register background cleanup task
        background_tasks.add_task(cleanup_job_directory, job_dir)
        
        # Return the generated PLY file
        return FileResponse(ply_path, media_type="application/octet-stream", filename=f"{job_id}.ply")
        
    except Exception as e:
        shutil.rmtree(job_dir, ignore_errors=True)
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/reconstruct-4d")
async def reconstruct_4d(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    landmarks: Optional[str] = Form(None)
):
    """
    Runs vitdet and 4D-Humans 3D mesh reconstruction on the uploaded image and returns the GLB file.
    """
    job_id = str(uuid.uuid4())
    job_dir = os.path.join(TEMP_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)
    
    image_path = os.path.join(job_dir, "image.png")
    glb_path = os.path.join(job_dir, "model.glb")
    
    try:
        # Pre-emptive VRAM cleanup to prevent CUDA Out Of Memory crashes
        print(">>> [VRAM] Reclaiming GPU memory before 4D-Humans reconstruction...")
        try:
            catvton_service.unload_models()
        except Exception:
            pass
        try:
            from upscaler_service import upscaler_service
            upscaler_service.unload_models()
        except Exception:
            pass
        try:
            from hmr2_service import hmr2_service
            hmr2_service.unload_models()
        except Exception:
            pass

        with open(image_path, "wb") as f:
            shutil.copyfileobj(image.file, f)
            
        import json
        parsed_landmarks = None
        if landmarks:
            try:
                parsed_landmarks = json.loads(landmarks)
            except Exception as e:
                print(f"[WARNING] Failed to parse landmarks JSON: {str(e)}")

        print(f">>> Running 4D-Humans reconstruction on GPU for job: {job_id}")
        from hmr2_service import hmr2_service
        success = hmr2_service.generate_hmr2_mesh(image_path, glb_path, landmarks=parsed_landmarks)
        
        if not success or not os.path.exists(glb_path):
            raise HTTPException(status_code=500, detail="4D-Humans GLB file was not generated.")
            
        # Post-generation VRAM cleanup
        try:
            hmr2_service.unload_models()
        except Exception as e:
            print(f"[VRAM ERROR] Failed to unload 4D-Humans: {str(e)}")

        # Register background cleanup task
        background_tasks.add_task(cleanup_job_directory, job_dir)
        
        # Return the generated GLB file
        return FileResponse(glb_path, media_type="model/gltf-binary", filename=f"{job_id}.glb")
        
    except Exception as e:
        shutil.rmtree(job_dir, ignore_errors=True)
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
