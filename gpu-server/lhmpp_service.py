import os
import subprocess

LHM_ROOT = os.environ.get("LHM_ROOT", "/home/bisagn/projectgsv/3dhuman/models/LHM-plusplus")

class LhmppService:
    def reconstruct_3d(self, job_id: str, image_path: str) -> str:
        # image_path is the try-on output image path
        # The output ply will be at LHM_ROOT/outputs/tpose_output/{job_id}.ply
        
        # Verify LHM++ exists
        if not os.path.exists(LHM_ROOT):
            raise FileNotFoundError(f"LHM++ project directory not found at {LHM_ROOT}")

        # Construct the command
        cmd = [
            "python", "scripts/inference/to_gs_ply.py",
            "--model_name", "LHMPP-700M-PixelShuffle",
            "--model_path", "./pretrained_models/Damo_XR_Lab/LHMPP-700M-PixelShuffle",
            "--image_glob", image_path
        ]
        
        print(f">>> Running LHM++ reconstruction for job: {job_id}...")
        print(f"Command: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            cwd=LHM_ROOT,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(">>> [ERROR] LHM++ execution failed!")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            raise RuntimeError(f"LHM++ execution failed: {result.stderr}")
            
        # Locate the output PLY file
        output_ply = os.path.join(LHM_ROOT, "outputs", "tpose_output", f"{job_id}.ply")
        if not os.path.exists(output_ply):
            # If default name logic falls back, list files to see if it saved under parent directory name
            # Normally we place image_path at /tmp/jobs/{job_id}/image.png, so parent is {job_id}
            # and it will save exactly under outputs/tpose_output/{job_id}.ply
            print(f">>> [WARNING] PLY file not found at expected path: {output_ply}")
            tpose_dir = os.path.join(LHM_ROOT, "outputs", "tpose_output")
            if os.path.exists(tpose_dir):
                files = os.listdir(tpose_dir)
                print(f"Contents of outputs/tpose_output/: {files}")
            raise FileNotFoundError(f"LHM++ output PLY file was not created for job {job_id}")
            
        print(f">>> [SUCCESS] 3D splat file created: {output_ply}")
        return output_ply

lhmpp_service = LhmppService()
