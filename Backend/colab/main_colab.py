import os
import argparse
import sys
import uuid
import uvicorn

# Prepend the colab directory to sys.path to prioritize colab/services/ and colab/utils/
# over standard backend folders when running the colab server.
colab_dir = os.path.dirname(os.path.abspath(__file__))
if colab_dir not in sys.path:
    sys.path.insert(0, colab_dir)

from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from services.trellis_service import trellis_service
from utils.tunnel import start_tunnel, stop_tunnel

app = FastAPI(
    title="4D-Humans VR - Colab Visual Generation Server",
    description="Provides heavy Image-to-3D visual mesh generation services using TRELLIS on Colab GPUs.",
    version="1.0.0"
)

# Enable CORS for requests coming from the local frontend or local backend proxy
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_DIR = "/content/temp_generation" if os.path.exists("/content") else "/kaggle/working/temp_generation"
os.makedirs(TEMP_DIR, exist_ok=True)

@app.post("/api/v1/colab/generate")
async def generate_colab_mesh(
    photo: UploadFile = File(..., description="The user portrait to reconstruct 3D body from")
):
    """
    Receives an image, removes the background, runs TRELLIS, and returns the generated GLB mesh directly.
    """
    # 1. Validate MIME type
    if photo.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: {photo.content_type}. Only JPEG and PNG are allowed."
        )

    job_id = str(uuid.uuid4())
    file_ext = os.path.splitext(photo.filename)[1] or ".jpg"
    
    input_path = os.path.join(TEMP_DIR, f"{job_id}_input{file_ext}")
    output_path = os.path.join(TEMP_DIR, f"{job_id}_mesh.glb")

    try:
        # Save uploaded file locally
        with open(input_path, "wb") as buffer:
            buffer.write(await photo.read())
            
        # Run TRELLIS mesh generation
        success = trellis_service.generate_mesh(input_path, output_path, remove_bg=True)
        
        # Cleanup input image
        if os.path.exists(input_path):
            os.remove(input_path)

        if not success or not os.path.exists(output_path):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="TRELLIS visual generation failed."
            )

        # Return the binary GLB file directly
        return FileResponse(
            path=output_path,
            filename=f"digital_twin_{job_id[:8]}.glb",
            media_type="model/gltf-binary"
        )

    except Exception as e:
        if os.path.exists(input_path):
            os.remove(input_path)
        if os.path.exists(output_path):
            os.remove(output_path)
            
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Colab generation server error: {str(e)}"
        )

@app.on_event("shutdown")
def on_shutdown():
    """
    Cleanup tunneling on shutdown.
    """
    stop_tunnel()

def main():
    parser = argparse.ArgumentParser(description="Start the Colab 3D Generation Server")
    parser.add_argument("--port", type=int, default=8000, help="Local port to run the server on")
    parser.add_argument("--ngrok-token", type=str, default=None, help="Ngrok authtoken to open a public tunnel")
    args = parser.parse_args()

    # Pre-initialize TRELLIS weights to speed up first request (runs heavy HuggingFace downloads)
    try:
        trellis_service.initialize_model()
    except Exception as e:
        print(f"[CRITICAL] Failed to initialize TRELLIS on startup: {str(e)}")
        print("Continuing startup anyway, but inference will fail if GPU is not loaded.")

    # Start ngrok tunnel if token is provided
    if args.ngrok_token:
        start_tunnel(args.port, args.ngrok_token)

    # Start FastAPI server
    try:
        uvicorn.run(app, host="127.0.0.1", port=args.port)
    except KeyboardInterrupt:
        print("\nStopping Colab server...")
    finally:
        stop_tunnel()

if __name__ == "__main__":
    main()
