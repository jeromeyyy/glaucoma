import os
import sys
import time
import webbrowser
import threading
import subprocess

def print_banner():
    banner = """
================================================================================
   GLAUCOMASCREEN-XAI: RESEARCH SCREENING & EXPLAINABILITY SYSTEM
   EfficientNet-B3 + Hypergraph Neural Networks (X-HGNN) + VLT
================================================================================
"""
    print(banner)

def check_dependencies():
    print("[1/4] Checking Python environment and dependencies...")
    required = ["torch", "torchvision", "flask", "albumentations", "cv2", "reportlab", "sklearn"]
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    
    if missing:
        print(f"[-] Missing dependencies: {missing}")
        print("[*] Installing missing dependencies from requirements.txt...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("[+] Dependencies successfully verified!")
    else:
        print("[+] All core dependencies are installed and verified.")

def check_data_and_checkpoints():
    print("[2/4] Checking dataset and model checkpoints...")
    from config import Config
    from datasets.sample_generator import create_synthetic_dataset
    from training.train import train_model

    # Ensure synthetic dataset exists for out-of-the-box operation
    synthetic_train_normal = os.path.join(Config.SYNTHETIC_DIR, "train", "Normal")
    if not os.path.exists(synthetic_train_normal) or len(os.listdir(synthetic_train_normal)) == 0:
        print("[*] Creating out-of-the-box synthetic fundus cohort...")
        create_synthetic_dataset(Config.SYNTHETIC_DIR)
        print("[+] Sample fundus dataset ready.")
    else:
        print("[+] Sample fundus dataset detected.")

    # Ensure checkpoint exists
    if not os.path.exists(Config.BEST_MODEL_PATH):
        print("[*] No checkpoint detected. Initializing and training 1 epoch for immediate readiness...")
        train_model(model_type="proposed", dataset_name="synthetic", num_epochs=1, batch_size=4)
        print("[+] Checkpoint created at checkpoints/best_model.pth")
    else:
        print(f"[+] Trained model checkpoint detected at {Config.BEST_MODEL_PATH}")

    # Ensure baseline checkpoint exists for ablation comparison
    if not os.path.exists(Config.BASELINE_MODEL_PATH):
        print("[*] Initializing baseline CNN checkpoint for ablation comparison...")
        train_model(model_type="baseline_cnn", dataset_name="synthetic", num_epochs=1, batch_size=4)
        print("[+] Baseline CNN checkpoint created at checkpoints/baseline_cnn.pth")
    else:
        print(f"[+] Baseline checkpoint detected at {Config.BASELINE_MODEL_PATH}")

def open_browser():
    time.sleep(2)
    url = "http://127.0.0.1:5000"
    print(f"\n[+] Opening web interface at {url} in your default browser...")
    webbrowser.open(url)

def start_server():
    print("[3/4] Launching Flask Application Server on http://127.0.0.1:5000...")
    print("[4/4] Starting medical dashboard services...")
    print("\n" + "-"*80)
    print(" >>> SYSTEM RUNNING! Press CTRL+C in this terminal to stop the application. <<<")
    print("-" * 80 + "\n")

    # Open browser in a separate thread
    threading.Thread(target=open_browser, daemon=True).start()

    from app import app
    app.run(host="0.0.0.0", port=5000, debug=False)

if __name__ == "__main__":
    print_banner()
    check_dependencies()
    check_data_and_checkpoints()
    start_server()
