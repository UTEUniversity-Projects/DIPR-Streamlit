# === RESUME TRAINING AFTER CRASH WITH DATASET SETUP ===

# 1. Cài đặt thư viện
!pip install ultralytics roboflow psutil

# 2. Import libraries
import os
import torch
from ultralytics import YOLO
from google.colab import files
from roboflow import Roboflow
import matplotlib.pyplot as plt
import cv2
import glob
import shutil
import yaml

# 3. Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 4. Setup dataset lại (bắt buộc)
print("Setting up dataset...")

# Download dataset từ Roboflow
rf = Roboflow(api_key="u8iCAFpURAQXlILez6iz")
project = rf.workspace("erovoutikaworkspace").project("fruits-24xta")
dataset = project.version(1).download("yolov8")

# Setup dataset directory
dataset_src = 'Fruits-1'
dataset_dest = '/content/datasets/Fruits-1'

# Tạo thư mục và copy dataset
os.makedirs('/content/datasets', exist_ok=True)
if os.path.exists(dataset_src) and not os.path.exists(dataset_dest):
    shutil.copytree(dataset_src, dataset_dest)
    print(f"Copied dataset to: {dataset_dest}")

# Fix data.yaml
data_yaml_path = os.path.join(dataset_dest, 'data.yaml')

if os.path.exists(data_yaml_path):
    # Load và sửa data.yaml
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    # Fix paths
    data_config['path'] = dataset_dest
    data_config['train'] = 'train/images'
    data_config['val'] = 'valid/images'
    data_config['test'] = 'test/images'
    
    # Ensure correct class names for fruits
    fruit_classes = ['Apple', 'Banana', 'Kiwi', 'Orange', 'Pear']
    data_config['names'] = fruit_classes
    data_config['nc'] = len(fruit_classes)
    
    # Save fixed data.yaml
    with open(data_yaml_path, 'w') as f:
        yaml.dump(data_config, f)
    
    print(f"✅ Dataset setup complete. data.yaml fixed at: {data_yaml_path}")
else:
    print(f"❌ data.yaml not found at: {data_yaml_path}")

# 5. Tìm checkpoint cuối cùng
checkpoint_path = '/content/drive/MyDrive/fruit_detection_project/fruit_detector_custom/weights/last.pt'

# Kiểm tra checkpoint tồn tại
if os.path.exists(checkpoint_path):
    print(f"✅ Found checkpoint: {checkpoint_path}")
    
    # 6. Load model từ checkpoint
    model = YOLO(checkpoint_path)
    print("✅ Model loaded successfully!")
    
    # 7. Resume training với data path chính xác
    print("\n🚀 Resuming training...")
    results = model.train(
        resume=True,
        data=data_yaml_path  # Chỉ định data path cụ thể
    )
    
    # 8. Validation sau khi training xong
    print("\nValidating model...")
    best_model_path = '/content/drive/MyDrive/fruit_detection_project/fruit_detector_custom/weights/best.pt'
    
    if os.path.exists(best_model_path):
        best_model = YOLO(best_model_path)
        metrics = best_model.val(data=data_yaml_path)
        
        print("\nEvaluation Results:")
        print(f"mAP50: {metrics.box.map50:.3f}")
        print(f"mAP50-95: {metrics.box.map:.3f}")
        print(f"Precision: {metrics.box.mp:.3f}")
        print(f"Recall: {metrics.box.mr:.3f}")
        
        # 9. Download model cuối cùng
        files.download(best_model_path)
        print(f"Downloaded: {best_model_path}")
    
else:
    print("❌ Checkpoint not found!")
    print("Please check Google Drive for backup files.")
    
    # Hiển thị các file có sẵn để debug
    if os.path.exists('/content/drive/MyDrive/fruit_detection_project'):
        print("\nAvailable files in project directory:")
        for root, dirs, files in os.walk('/content/drive/MyDrive/fruit_detection_project'):
            for file in files:
                print(os.path.join(root, file))