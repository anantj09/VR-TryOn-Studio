import os
import sys
import subprocess
import shutil

def run_command(args, cwd=None):
    cmd_str = ' '.join(args)
    print(f"\n>>> Running: {cmd_str} in {cwd or 'root'}")
    # Use shell=True to support virtualenv scripts/executables across Windows and Linux
    result = subprocess.run(args, cwd=cwd, shell=True)
    if result.returncode != 0:
        print(f"\n[ERROR] Command failed with return code {result.returncode}: {cmd_str}")
        sys.exit(result.returncode)

def main():
    print("==========================================================")
    print("        VR Try-On Studio: Automated Setup Script")
    print("==========================================================")
    
    project_root = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.join(project_root, "Backend")
    gpu_dir = os.path.join(project_root, "gpu-server")
    
    # 1. Create essential data directories
    data_dirs = [
        os.path.join(project_root, "data", "meshes"),
        os.path.join(project_root, "data", "tryon_outputs"),
        os.path.join(project_root, "data", "clothing"),
        os.path.join(project_root, "data", "premade"),
    ]
    for d in data_dirs:
        if not os.path.exists(d):
            os.makedirs(d)
            print(f"Created data directory: {os.path.basename(d)}")

    # 2. Setup CPU Backend Virtual Environment
    print("\n--- [1/2] Setting up CPU Backend Virtual Environment ---")
    backend_venv = os.path.join(backend_dir, ".venv")
    if not os.path.exists(backend_venv):
        print("Creating virtual environment in Backend/.venv...")
        run_command([sys.executable, "-m", "venv", ".venv"], cwd=backend_dir)
    else:
        print("Backend virtual environment already exists.")

    pip_bin = os.path.join(backend_venv, "Scripts", "pip") if os.name == 'nt' else os.path.join(backend_venv, "bin", "pip")
    run_command([pip_bin, "install", "--upgrade", "pip"], cwd=backend_dir)
    run_command([pip_bin, "install", "-r", "requirements.txt"], cwd=backend_dir)

    # 3. Setup GPU Server Virtual Environment (Optional)
    print("\n--- [2/2] Setting up GPU Server Virtual Environment ---")
    print("👉 NOTE: The GPU Compute Server runs heavy AI models (CatVTON try-on, LHM++ splatting, 4D-Humans).")
    print("   Requirements: NVIDIA GPU (>= 8GB VRAM, 12GB+ recommended) and ~25GB of free disk space for model weights.")
    setup_gpu = input("Do you want to setup a local virtual env for the GPU Server? (y/N): ")
    if setup_gpu.lower() == 'y':
        gpu_venv = os.path.join(gpu_dir, ".venv")
        if not os.path.exists(gpu_venv):
            print("Creating virtual environment in gpu-server/.venv...")
            run_command([sys.executable, "-m", "venv", ".venv"], cwd=gpu_dir)
        else:
            print("GPU server virtual environment already exists.")

        gpu_pip = os.path.join(gpu_venv, "Scripts", "pip") if os.name == 'nt' else os.path.join(gpu_venv, "bin", "pip")
        run_command([gpu_pip, "install", "--upgrade", "pip"], cwd=gpu_dir)
        
        # Install CUDA-compatible PyTorch first (matching GPU driver CUDA version)
        cuda_option = input("Select PyTorch CUDA version:\n1. CUDA 11.8 (Recommended for older GPUs/Ubuntu)\n2. CUDA 12.1 (Recommended for RTX 30/40 series)\n3. CPU Only (Testing only)\nChoice [1/2/3]: ")
        if cuda_option == '1':
            print("Installing PyTorch with CUDA 11.8...")
            run_command([gpu_pip, "install", "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu118"], cwd=gpu_dir)
        elif cuda_option == '2':
            print("Installing PyTorch with CUDA 12.1...")
            run_command([gpu_pip, "install", "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu121"], cwd=gpu_dir)
        else:
            print("Installing PyTorch CPU version...")
            run_command([gpu_pip, "install", "torch", "torchvision"], cwd=gpu_dir)

        # Install remainder of GPU dependencies (use --no-build-isolation for compilation of basicsr/detectron2)
        print("Installing GPU server dependencies (this may take a few minutes)...")
        run_command([gpu_pip, "install", "-r", "requirements.txt", "--no-build-isolation"], cwd=gpu_dir)

        # 4. Optional: Download AI Model Weights
        print("\n--- [3/3] Optional AI Model Weights Pre-downloading ---")
        download_weights = input("Do you want to pre-download all AI model weights (CatVTON, HMR2/ViTDet, Real-ESRGAN, GFPGAN, LHM++ models/priors)? (y/N): ")
        if download_weights.lower() == 'y':
            gpu_python = os.path.join(gpu_venv, "Scripts", "python") if os.name == 'nt' else os.path.join(gpu_venv, "bin", "python")
            run_command([gpu_python, "setup_models.py"], cwd=gpu_dir)

    print("\n==========================================================")
    print("            [SETUP COMPLETED SUCCESSFULY]")
    print("==========================================================")
    print("- To start the Backend API: cd Backend && .venv\\Scripts\\activate && uvicorn main:app --reload")
    print("- To run the Frontend server: python -m http.server 8080 --directory frontend")
    if setup_gpu.lower() == 'y':
        print("- To run the GPU server locally: cd gpu-server && .venv\\Scripts\\activate && python main.py")
    print("- Or run everything using Docker: docker-compose up --build")
    print("==========================================================")

if __name__ == "__main__":
    main()
