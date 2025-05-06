# ===============================================
# FIXED CODE: Training YOLOv8 for Fruit Detection on Google Colab
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
import psutil
import time
import threading
from IPython.display import clear_output
import random
import cv2
import numpy as np

# 2. MOUNT GOOGLE DRIVE - Để lưu checkpoints
from google.colab import drive
drive.mount('/content/drive')

# Tạo thư mục cho project
project_dir = '/content/drive/MyDrive/fruit_detection_project'
os.makedirs(project_dir, exist_ok=True)

# 3. CHECK RESOURCES - Kiểm tra tài nguyên
def check_resources():
    print("=== RESOURCE CHECK ===")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU'}")
    print(f"RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    print(f"RAM Available: {psutil.virtual_memory().available / (1024**3):.1f} GB")
    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
    print("===================")

check_resources()

# 4. DOWNLOAD DATASET - Tải dataset từ Roboflow
rf = Roboflow(api_key="u8iCAFpURAQXlILez6iz")
project = rf.workspace("erovoutikaworkspace").project("fruits-24xta")
dataset_download = project.version(1).download("yolov8")

# 5. SETUP DATASET DIRECTORY - Chuẩn bị thư mục dataset
dataset_src = 'Fruits-1'
dataset_dest = '/content/datasets/Fruits-1'

# Tạo thư mục đích
os.makedirs('/content/datasets', exist_ok=True)

# Copy dataset
if os.path.exists(dataset_src) and not os.path.exists(dataset_dest):
    shutil.copytree(dataset_src, dataset_dest)
    print(f"Copied dataset to: {dataset_dest}")

# 6. FIX DATA.YAML - Sửa file cấu hình dataset
data_yaml_path = os.path.join(dataset_dest, 'data.yaml')

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

print("\nFixed data.yaml:")
print(yaml.dump(data_config, sort_keys=False))

# 7. VERIFY DATASET - Kiểm tra dataset
def verify_dataset(data_yaml_path):
    """Verify dataset structure and content"""
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    print("\n=== DATASET VERIFICATION ===")
    print(f"Classes: {data_config['names']}")
    print(f"Number of classes: {data_config['nc']}")
    
    # Check each split
    for split in ['train', 'val', 'test']:
        if split in data_config:
            img_path = os.path.join(data_config['path'], data_config[split])
            label_path = img_path.replace('images', 'labels')
            
            img_count = len([f for f in os.listdir(img_path) if f.endswith(('.jpg', '.png'))])
            label_count = len([f for f in os.listdir(label_path) if f.endswith('.txt')])
            
            print(f"\n{split.upper()} set:")
            print(f"  Images: {img_count}")
            print(f"  Labels: {label_count}")
            
            # Check a random image and its label
            if img_count > 0:
                random_img = random.choice([f for f in os.listdir(img_path) if f.endswith('.jpg')])
                img_file = os.path.join(img_path, random_img)
                label_file = os.path.join(label_path, random_img.replace('.jpg', '.txt'))
                
                print(f"  Sample image: {random_img}")
                if os.path.exists(label_file):
                    with open(label_file, 'r') as f:
                        labels = f.readlines()
                        print(f"  Labels found: {len(labels)}")
                        if labels:
                            print(f"  First label: {labels[0].strip()}")

verify_dataset(data_yaml_path)

# 8. VISUALIZE DATASET - Kiểm tra ảnh và label
def visualize_dataset(data_yaml_path, num_samples=3):
    """Visualize some training samples"""
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    train_img_path = os.path.join(data_config['path'], data_config['train'])
    train_label_path = train_img_path.replace('images', 'labels')
    
    images = [f for f in os.listdir(train_img_path) if f.endswith('.jpg')]
    selected_images = random.sample(images, min(num_samples, len(images)))
    
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 5))
    if num_samples == 1:
        axes = [axes]
    
    for idx, img_name in enumerate(selected_images):
        # Load image
        img_path = os.path.join(train_img_path, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Load labels
        label_path = os.path.join(train_label_path, img_name.replace('.jpg', '.txt'))
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                labels = f.readlines()
            
            # Draw bounding boxes
            h, w = img.shape[:2]
            for label in labels:
                class_id, x_center, y_center, width, height = map(float, label.split())
                class_name = data_config['names'][int(class_id)]
                
                # Convert to pixel coordinates
                x1 = int((x_center - width/2) * w)
                y1 = int((y_center - height/2) * h)
                x2 = int((x_center + width/2) * w)
                y2 = int((y_center + height/2) * h)
                
                # Draw rectangle and label
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, class_name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        axes[idx].imshow(img)
        axes[idx].axis('off')
        axes[idx].set_title(f"Sample {idx+1}")
    
    plt.tight_layout()
    plt.show()

visualize_dataset(data_yaml_path)

# 9. CREATE CUSTOM YOLO CONFIG - Tạo config cho model
def create_custom_config(num_classes):
    """Create custom YOLOv8 config for fruit detection"""
    config = {
        'nc': num_classes,
        'depth_multiple': 1.0,
        'width_multiple': 1.0,
        'backbone': [
            [-1, 1, 'Conv', [64, 3, 2]],
            [-1, 1, 'Conv', [128, 3, 2]],
            [-1, 3, 'C2f', [128, True]],
            [-1, 1, 'Conv', [256, 3, 2]],
            [-1, 6, 'C2f', [256, True]],
            [-1, 1, 'Conv', [512, 3, 2]],
            [-1, 6, 'C2f', [512, True]],
            [-1, 1, 'Conv', [1024, 3, 2]],
            [-1, 3, 'C2f', [1024, True]],
            [-1, 1, 'SPPF', [1024, 5]],
        ],
        'head': [
            [-1, 1, 'Conv', [1024, 3, 1]],
            [-1, 1, 'Conv', [1024, 3, 1]],
            [[15, 18, 21], 1, 'Detect', [num_classes]]
        ]
    }
    return config

# 10. TRAINING SETUP - Cấu hình training
# Create a new model from scratch for fruit detection
model = YOLO()  # Create empty model
model.model = YOLO('yolov8n.yaml').model  # Load architecture
model.model.nc = 5  # Set number of classes to 5 for fruits

# Or better: create custom model for fruits
#custom_config = create_custom_config(5)
#with open('custom_fruit_model.yaml', 'w') as f:
#    yaml.dump(custom_config, f)
#model = YOLO('custom_fruit_model.yaml')

training_params = {
    'data': data_yaml_path,
    'epochs': 50,
    'imgsz': 640,
    'batch': 16,
    'optimizer': 'AdamW',
    'lr0': 0.01,  # Higher learning rate for training from scratch
    'lrf': 0.1,
    'weight_decay': 0.005,
    'warmup_epochs': 10,
    'warmup_momentum': 0.8,
    'save_period': 10,
    'patience': 20,
    'cache': True,
    'workers': 4,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'name': 'fruit_detector_custom',
    'project': project_dir,
    'exist_ok': True,
    'resume': False,  # Start fresh
    'pretrained': False  # No pretrained weights
}

print("\nTraining parameters:")
for key, value in training_params.items():
    print(f"{key}: {value}")

# 11. START TRAINING - Bắt đầu training
print("\nStarting training...")
try:
    results = model.train(**training_params)
except Exception as e:
    print(f"Training error: {e}")
    # Save checkpoint even on error
    model.save(os.path.join(project_dir, 'checkpoint_error.pt'))

# 12. VALIDATION - Kiểm tra kết quả
print("\nValidating model...")
best_model_path = os.path.join(project_dir, 'fruit_detector_custom/weights/best.pt')

if os.path.exists(best_model_path):
    best_model = YOLO(best_model_path)
    metrics = best_model.val(data=data_yaml_path)
    
    print("\nEvaluation Results:")
    print(f"mAP50: {metrics.box.map50:.3f}")
    print(f"mAP50-95: {metrics.box.map:.3f}")
    print(f"Precision: {metrics.box.mp:.3f}")
    print(f"Recall: {metrics.box.mr:.3f}")
    
    # 13. TEST ON SAMPLE IMAGES
    print("\nTesting on sample images:")
    import glob
    
    val_img_path = os.path.join(dataset_dest, 'valid/images')
    val_images = glob.glob(f'{val_img_path}/*')[:3]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    if len(val_images) == 1:
        axes = [axes]
    elif len(val_images) == 2:
        axes = axes[:2]
    
    for idx, img_path in enumerate(val_images):
        results = best_model(img_path)
        annotated_frame = results[0].plot()
        
        axes[idx].imshow(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
        axes[idx].axis('off')
        axes[idx].set_title(os.path.basename(img_path))
    
    plt.tight_layout()
    plt.show()
    
    # 14. SAVE AND DOWNLOAD MODEL
    final_model_path = os.path.join(project_dir, 'fruit_detection_final.pt')
    best_model.save(final_model_path)
    print(f"Saved final model to: {final_model_path}")
    
    from google.colab import files
    files.download(best_model_path)
    print(f"Downloaded: {best_model_path}")
else:
    print("Best model not found. Training may have failed.")

# 15. CLEANUP
import gc
gc.collect()
torch.cuda.empty_cache()

print("""
TRAINING COMPLETE!

1. Download the 'best.pt' file
2. Rename it to 'fruit_detection.pt'
3. Place it in the 'models/' directory of your Streamlit project
4. Run your Streamlit app: streamlit run app.py

The model should now correctly detect fruits (Apple, Banana, Kiwi, Orange, Pear)
""")