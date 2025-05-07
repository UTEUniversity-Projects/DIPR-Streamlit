#!/usr/bin/env python3
# ===============================================
# Training YOLOv8 for Animal Detection on Google Colab
# With Google Drive checkpoint support for crash recovery
# ===============================================

# 1. SETUP - Cài đặt thư viện
!pip install ultralytics roboflow psutil

# Import libraries
import os
import torch
from ultralytics import YOLO
from roboflow import Roboflow
import matplotlib.pyplot as plt
import shutil
import yaml
import signal
import sys
import time

# 2. MOUNT GOOGLE DRIVE - Mount Drive first for checkpoint saving
from google.colab import drive
drive.mount('/content/drive')

# Create project directory in Drive
PROJECT_DIR = '/content/drive/MyDrive/animal_detection_project'
os.makedirs(PROJECT_DIR, exist_ok=True)

# 3. EMERGENCY SAVE HANDLER
def emergency_save(signum, frame):
    """Emergency save when crash is detected"""
    print("\n🚨 CRASH DETECTED - SAVING MODEL...")
    try:
        emergency_path = os.path.join(PROJECT_DIR, 'emergency_checkpoint.pt')
        model.save(emergency_path)
        print(f"✅ Emergency save to: {emergency_path}")
    except Exception as e:
        print(f"❌ Emergency save failed: {e}")
    sys.exit(0)

# Register crash handler
signal.signal(signal.SIGTERM, emergency_save)
signal.signal(signal.SIGINT, emergency_save)

# 4. KEEP ALIVE FUNCTION (prevent session timeout)
def keep_alive():
    """Prevent idle timeout"""
    while True:
        from IPython.display import clear_output
        clear_output(wait=True)
        print("Keep alive - current time:", time.ctime())
        time.sleep(300)  # 5 minutes

# Start keep-alive thread
import threading
alive_thread = threading.Thread(target=keep_alive)
alive_thread.daemon = True
alive_thread.start()

# 5. DOWNLOAD DATASET - Tải dataset từ Roboflow
print("Downloading dataset from Roboflow...")
rf = Roboflow(api_key="u8iCAFpURAQXlILez6iz")
project = rf.workspace("animal-detection-using-yolov8").project("animal-detection-using-yolov8")
version = project.version(5)
dataset = version.download("yolov8")

# 6. SETUP DATASET DIRECTORY - Chuẩn bị thư mục dataset
# Kiểm tra cấu trúc hiện tại
print("Current directory:", os.getcwd())
print("Contents:", os.listdir())

# Đảm bảo dataset được đặt đúng chỗ
dataset_src = 'animal-detection-using-yolov8-5'
dataset_dest = '/content/datasets/animal-detection-using-yolov8-5'

# Tạo thư mục đích
os.makedirs('/content/datasets', exist_ok=True)

# Copy dataset đến vị trí mà Ultralytics mong đợi
if os.path.exists(dataset_src) and not os.path.exists(dataset_dest):
    shutil.copytree(dataset_src, dataset_dest)
    print(f"Copied dataset to: {dataset_dest}")
elif os.path.exists(dataset_src):
    print(f"Dataset already exists at: {dataset_dest}")
else:
    print("Dataset directory not found!")

# 7. FIX DATA.YAML - Sửa file cấu hình dataset
data_yaml_path = os.path.join(dataset_dest, 'data.yaml')

# Đảm bảo file data.yaml tồn tại
if os.path.exists(data_yaml_path):
    # Đọc và sửa data.yaml
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    # Fix paths
    data_config['path'] = dataset_dest
    data_config['train'] = 'train/images'
    data_config['val'] = 'valid/images'
    data_config['test'] = 'test/images'
    
    # Ghi lại file
    with open(data_yaml_path, 'w') as f:
        yaml.dump(data_config, f)
    
    print("Updated data.yaml:")
    print(yaml.dump(data_config))
else:
    print(f"Warning: data.yaml not found at {data_yaml_path}")
    print("Creating a new data.yaml file...")
    
    # Create a basic data.yaml if it doesn't exist
    data_config = {
        'path': dataset_dest,
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'nc': 10,  # Adjust based on the actual number of classes
        'names': ['dog', 'cat', 'bird', 'horse', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'other']  # Adjust class names
    }
    
    # Write to file
    with open(data_yaml_path, 'w') as f:
        yaml.dump(data_config, f)
    
    print("Created new data.yaml:")
    print(yaml.dump(data_config))

# 8. VERIFY DATASET - Kiểm tra dataset
print("\nVerifying dataset structure:")
with open(data_yaml_path, 'r') as f:
    data = yaml.safe_load(f)
    print("Data yaml content:")
    print(data)
    
    for split in ['train', 'val', 'test']:
        if split in data:
            full_path = os.path.join(data.get('path', dataset_dest), data[split])
            exists = os.path.exists(full_path)
            print(f"{split} path: {full_path} - Exists: {exists}")
            
            if exists:
                files = [f for f in os.listdir(full_path) if f.endswith(('.jpg', '.png'))]
                print(f"  Number of images: {len(files)}")

# 9. CHECK FOR EXISTING CHECKPOINT
checkpoint_path = os.path.join(PROJECT_DIR, 'animal_detector/weights/last.pt')
resume_training = os.path.exists(checkpoint_path)

if resume_training:
    print(f"\n✅ Found checkpoint at: {checkpoint_path}")
    print("Resuming from previous training...")
    model = YOLO(checkpoint_path)
else:
    print("\n⚠️ No checkpoint found. Starting fresh training...")
    model = YOLO('yolov8n.pt')  # YOLOv8 nano model

# 10. TRAINING SETUP - Cấu hình training
training_params = {
    'data': data_yaml_path,
    'epochs': 100,
    'imgsz': 640,
    'batch': 16,  # Giảm batch nếu GPU nhỏ
    'optimizer': 'AdamW',
    'lr0': 0.001,
    'lrf': 0.1,
    'weight_decay': 0.005,
    'warmup_epochs': 5,
    'warmup_momentum': 0.8,
    'save_period': 5,  # Save every 5 epochs
    'patience': 15,
    'cache': True,
    'workers': 8,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'name': 'animal_detector',
    'project': PROJECT_DIR,  # Save to Google Drive
    'exist_ok': True,
    'resume': resume_training  # Resume if checkpoint exists
}

print("\nTraining parameters:")
for key, value in training_params.items():
    print(f"{key}: {value}")

# 11. START TRAINING - Bắt đầu training
print("\nStarting training...")
try:
    results = model.train(**training_params)
except KeyboardInterrupt:
    print("\nTraining interrupted by user")
    emergency_save(None, None)
except Exception as e:
    print(f"\nTraining error: {e}")
    # Save checkpoint on error
    error_checkpoint = os.path.join(PROJECT_DIR, 'error_checkpoint.pt')
    model.save(error_checkpoint)
    print(f"Saved checkpoint at: {error_checkpoint}")

# 12. VALIDATION - Kiểm tra kết quả
print("\nValidating model...")
best_model_path = os.path.join(PROJECT_DIR, 'animal_detector/weights/best.pt')

if os.path.exists(best_model_path):
    best_model = YOLO(best_model_path)
    metrics = best_model.val(data=data_yaml_path)

    print("\nEvaluation Results:")
    print(f"mAP50: {metrics.box.map50:.3f}")
    print(f"mAP50-95: {metrics.box.map:.3f}")
    print(f"Precision: {metrics.box.mp:.3f}")
    print(f"Recall: {metrics.box.mr:.3f}")

# 13. TEST ON SAMPLE IMAGES - Thử nghiệm trên ảnh mẫu
import glob

print("\nTesting on sample images:")
val_images = glob.glob(f'{dataset_dest}/valid/images/*')[:3]  # Test 3 images

if val_images:
    fig, axes = plt.subplots(1, len(val_images), figsize=(15, 5))
    if len(val_images) == 1:
        axes = [axes]
    
    for idx, img_path in enumerate(val_images):
        results = best_model(img_path)
        annotated_frame = results[0].plot()
        
        axes[idx].imshow(annotated_frame)
        axes[idx].axis('off')
        axes[idx].set_title(os.path.basename(img_path))
    
    plt.tight_layout()
    plt.show()

# 14. EXPORT MODEL - Xuất model
print("\nExporting model...")
# Export to PyTorch format
final_model_path = os.path.join(PROJECT_DIR, 'animal_detection_final.pt')
best_model.save(final_model_path)

# Optional: Export to ONNX format
#model.export(format='onnx')

# 15. DOWNLOAD MODEL - Tải model về
from google.colab import files

# Download best model weights
if os.path.exists(best_model_path):
    files.download(best_model_path)
    print(f"Downloaded: {best_model_path}")

# Download final model
files.download(final_model_path)
print(f"Downloaded: {final_model_path}")

# Download training results
results_img = os.path.join(PROJECT_DIR, 'animal_detector/results.png')
if os.path.exists(results_img):
    files.download(results_img)
    print(f"Downloaded: {results_img}")

confusion_matrix_img = os.path.join(PROJECT_DIR, 'animal_detector/confusion_matrix.png')
if os.path.exists(confusion_matrix_img):
    files.download(confusion_matrix_img)
    print(f"Downloaded: {confusion_matrix_img}")

print("\nTraining complete! All models downloaded successfully.")

# 16. FINAL NOTES - Ghi chú cuối
print("""
FINAL STEPS:
1. Download the 'best.pt' file to your local computer
2. Rename it to 'animal_detection.pt'
3. Place it in the 'models/' directory of your Streamlit project
4. Run your Streamlit app: streamlit run app.py
5. Test the animal detection feature in your web app
""")