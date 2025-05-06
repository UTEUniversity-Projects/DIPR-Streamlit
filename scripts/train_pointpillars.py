#!/usr/bin/env python3
"""
Script to train PointPillars model on KITTI dataset and export to ONNX
Requires:
- PyTorch
- ONNX
- CUDA (GPU support)
- Open3D
- NumPy
- tqdm
"""

import os
import sys
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import onnx
import onnxruntime as ort
from tqdm import tqdm
import time

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Constants for KITTI dataset
KITTI_CLASSES = {
    'Car': 0,
    'Pedestrian': 1,
    'Cyclist': 2
}

# Point cloud ranges and voxel sizes for PointPillars
POINT_CLOUD_RANGE = [-75.2, -75.2, -2, 75.2, 75.2, 4]
VOXEL_SIZE = [0.16, 0.16, 4]
MAX_POINTS_PER_VOXEL = 100
MAX_VOXELS = 12000
BATCH_SIZE = 4  # Adjust based on your GPU memory
EPOCHS = 80
LEARNING_RATE = 0.001

class PillarFeatureNet(nn.Module):
    """Pillar Feature Network for PointPillars"""
    def __init__(self, input_channels=4, output_channels=64):
        super(PillarFeatureNet, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        
        # Define network layers
        self.conv1 = nn.Conv1d(input_channels, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, output_channels, 1)
        self.bn3 = nn.BatchNorm1d(output_channels)
        self.relu = nn.ReLU()
        
    def forward(self, voxels, num_points, coords):
        """
        Args:
            voxels: (B, N, P, C) tensor containing features of points in voxels
            num_points: (B, N) tensor containing number of points in each voxel
            coords: (B, N, 3) tensor containing voxel coordinates
        Returns:
            Tensor of shape (B, C, H, W) containing pillar features
        """
        batch_size = voxels.shape[0]
        voxel_count = voxels.shape[1]
        
        # Reshape for feature extraction
        features = voxels.view(-1, voxels.shape[2], self.input_channels)
        
        # Apply network
        features = self.relu(self.bn1(self.conv1(features.transpose(1, 2))))
        features = self.relu(self.bn2(self.conv2(features)))
        features = self.relu(self.bn3(self.conv3(features)))
        
        # Max pooling over points in each voxel
        features = torch.max(features, dim=2)[0]
        
        # Reshape back to batch format
        features = features.view(batch_size, voxel_count, self.output_channels)
        
        return features

class RegionProposalNetwork(nn.Module):
    """Region Proposal Network for PointPillars"""
    def __init__(self, in_channels=64, num_classes=3):
        super(RegionProposalNetwork, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        # Define network layers
        self.conv1 = nn.Conv2d(in_channels, 128, 3, 2, 1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 128, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(128)
        
        # Output layers for detection
        self.cls_head = nn.Conv2d(128, num_classes, 1)
        self.loc_head = nn.Conv2d(128, 7, 1)  # x, y, z, w, l, h, theta
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) tensor containing pillar features
        Returns:
            cls_preds: (B, num_classes, H, W) tensor containing class predictions
            loc_preds: (B, 7, H, W) tensor containing location predictions
        """
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        
        cls_preds = self.cls_head(x)
        loc_preds = self.loc_head(x)
        
        return cls_preds, loc_preds

class PointPillars(nn.Module):
    """Complete PointPillars model"""
    def __init__(self, input_channels=4, pfe_output_channels=64, num_classes=3):
        super(PointPillars, self).__init__()
        self.pfe = PillarFeatureNet(input_channels, pfe_output_channels)
        self.rpn = RegionProposalNetwork(pfe_output_channels, num_classes)
        
    def forward(self, voxels, num_points, coords):
        """
        Args:
            voxels: (B, N, P, C) tensor containing features of points in voxels
            num_points: (B, N) tensor containing number of points in each voxel
            coords: (B, N, 3) tensor containing voxel coordinates
        Returns:
            cls_preds: (B, num_classes, H, W) tensor containing class predictions
            loc_preds: (B, 7, H, W) tensor containing location predictions
        """
        # Get pillar features
        pillar_features = self.pfe(voxels, num_points, coords)
        
        # Scatter pillars to a 2D grid
        batch_size = voxels.shape[0]
        spatial_features = torch.zeros(
            (batch_size, pillar_features.shape[2], 
             int((POINT_CLOUD_RANGE[3] - POINT_CLOUD_RANGE[0]) / VOXEL_SIZE[0]),
             int((POINT_CLOUD_RANGE[4] - POINT_CLOUD_RANGE[1]) / VOXEL_SIZE[1])),
            device=voxels.device
        )
        
        # This is a simplified scatter - in reality you need a proper scatter operation
        # For demonstration, we'll randomly place features on the grid
        for b in range(batch_size):
            indices = torch.randint(
                0, spatial_features.shape[2] * spatial_features.shape[3], 
                (pillar_features.shape[1],)
            )
            x_idx = indices % spatial_features.shape[2]
            y_idx = indices // spatial_features.shape[2]
            spatial_features[b, :, x_idx, y_idx] = pillar_features[b].T
        
        # Run RPN
        cls_preds, loc_preds = self.rpn(spatial_features)
        
        return cls_preds, loc_preds

class KittiDataset(Dataset):
    """Dataset class for KITTI"""
    def __init__(self, data_root, split='train'):
        self.data_root = data_root
        self.split = split
        
        # Get file paths
        if split == 'train':
            self.lidar_dir = os.path.join(data_root, 'training', 'velodyne')
            self.label_dir = os.path.join(data_root, 'training', 'label_2')
            self.calib_dir = os.path.join(data_root, 'training', 'calib')
        else:
            self.lidar_dir = os.path.join(data_root, 'testing', 'velodyne')
            self.calib_dir = os.path.join(data_root, 'testing', 'calib')
            self.label_dir = None
        
        # Get file names
        self.sample_ids = [f.split('.')[0] for f in os.listdir(self.lidar_dir)]
        
    def __len__(self):
        return len(self.sample_ids)
    
    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        
        # Load point cloud
        lidar_file = os.path.join(self.lidar_dir, f"{sample_id}.bin")
        point_cloud = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)
        
        # Load labels if in training mode
        labels = None
        if self.split == 'train' and self.label_dir is not None:
            label_file = os.path.join(self.label_dir, f"{sample_id}.txt")
            labels = self._read_labels(label_file)
        
        # Process point cloud to voxels
        voxels, num_points, coords, labels_encoded = self._process_pointcloud(point_cloud, labels)
        
        return {
            'sample_id': sample_id,
            'point_cloud': point_cloud,
            'voxels': voxels,
            'num_points': num_points,
            'coords': coords,
            'labels': labels_encoded
        }
    
    def _read_labels(self, label_file):
        """Read labels from KITTI label file"""
        labels = []
        with open(label_file, 'r') as f:
            for line in f.readlines():
                parts = line.split()
                obj_type = parts[0]
                if obj_type not in KITTI_CLASSES:
                    continue
                
                bbox = [float(p) for p in parts[4:8]]
                dimensions = [float(p) for p in parts[8:11]]
                location = [float(p) for p in parts[11:14]]
                rotation = float(parts[14])
                
                labels.append({
                    'type': obj_type,
                    'class_id': KITTI_CLASSES[obj_type],
                    'bbox': bbox,
                    'dimensions': dimensions,
                    'location': location,
                    'rotation': rotation
                })
        
        return labels
    
    def _process_pointcloud(self, point_cloud, labels):
        """Convert point cloud to voxels for PointPillars"""
        # Filter points within range
        mask = (
            (point_cloud[:, 0] >= POINT_CLOUD_RANGE[0]) &
            (point_cloud[:, 0] < POINT_CLOUD_RANGE[3]) &
            (point_cloud[:, 1] >= POINT_CLOUD_RANGE[1]) &
            (point_cloud[:, 1] < POINT_CLOUD_RANGE[4]) &
            (point_cloud[:, 2] >= POINT_CLOUD_RANGE[2]) &
            (point_cloud[:, 2] < POINT_CLOUD_RANGE[5])
        )
        point_cloud = point_cloud[mask]
        
        # Simplified voxelization
        # In a real implementation, you would use a more efficient algorithm
        voxels = np.zeros((MAX_VOXELS, MAX_POINTS_PER_VOXEL, 4), dtype=np.float32)
        num_points = np.zeros(MAX_VOXELS, dtype=np.int32)
        coords = np.zeros((MAX_VOXELS, 3), dtype=np.int32)
        
        # Calculate voxel indices
        voxel_indices = (
            (point_cloud[:, :3] - np.array([POINT_CLOUD_RANGE[0], POINT_CLOUD_RANGE[1], POINT_CLOUD_RANGE[2]])) / 
            np.array(VOXEL_SIZE)
        ).astype(np.int32)
        
        # Simple aggregation - using a dictionary to group points by voxel
        voxel_dict = {}
        for i, (point, idx) in enumerate(zip(point_cloud, voxel_indices)):
            voxel_key = tuple(idx)
            if voxel_key not in voxel_dict:
                voxel_dict[voxel_key] = []
            voxel_dict[voxel_key].append(point)
        
        # Fill in voxels data
        for i, (voxel_key, points) in enumerate(voxel_dict.items()):
            if i >= MAX_VOXELS:
                break
                
            # Get points for this voxel
            voxel_points = np.array(points[:MAX_POINTS_PER_VOXEL])
            point_count = len(voxel_points)
            
            # Store data
            voxels[i, :point_count] = voxel_points
            num_points[i] = point_count
            coords[i] = np.array(voxel_key)
        
        # Encode labels for training
        # In a real implementation, you would convert labels to targets
        if labels is not None:
            # Placeholder for label encoding
            labels_encoded = {
                'cls_targets': np.zeros((100, 3), dtype=np.float32),  # Class targets
                'loc_targets': np.zeros((100, 7), dtype=np.float32)   # Location targets
            }
        else:
            labels_encoded = None
        
        return voxels, num_points, coords, labels_encoded

def train(model, train_dataloader, val_dataloader, optimizer, scheduler, num_epochs, device, output_dir):
    """Training function"""
    # Loss functions
    cls_loss_fn = nn.BCEWithLogitsLoss()
    loc_loss_fn = nn.SmoothL1Loss()
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_cls_loss = 0.0
        train_loc_loss = 0.0
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        progress_bar = tqdm(train_dataloader)
        
        for batch in progress_bar:
            voxels = batch['voxels'].to(device)
            num_points = batch['num_points'].to(device)
            coords = batch['coords'].to(device)
            
            # Forward pass
            cls_preds, loc_preds = model(voxels, num_points, coords)
            
            # This is simplified - in a real implementation you'd have proper targets
            cls_targets = torch.zeros_like(cls_preds)
            loc_targets = torch.zeros_like(loc_preds)
            
            # Calculate loss
            cls_loss = cls_loss_fn(cls_preds, cls_targets)
            loc_loss = loc_loss_fn(loc_preds, loc_targets)
            loss = cls_loss + loc_loss
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update stats
            train_loss += loss.item()
            train_cls_loss += cls_loss.item()
            train_loc_loss += loc_loss.item()
            
            # Update progress bar
            progress_bar.set_description(f"Loss: {loss.item():.4f}")
        
        # Scheduler step
        scheduler.step()
        
        # Calculate average losses
        train_loss /= len(train_dataloader)
        train_cls_loss /= len(train_dataloader)
        train_loc_loss /= len(train_dataloader)
        
        print(f"Train Loss: {train_loss:.4f}, CLS: {train_cls_loss:.4f}, LOC: {train_loc_loss:.4f}")
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validation"):
                voxels = batch['voxels'].to(device)
                num_points = batch['num_points'].to(device)
                coords = batch['coords'].to(device)
                
                # Forward pass
                cls_preds, loc_preds = model(voxels, num_points, coords)
                
                # Simplified loss calculation
                cls_targets = torch.zeros_like(cls_preds)
                loc_targets = torch.zeros_like(loc_preds)
                
                cls_loss = cls_loss_fn(cls_preds, cls_targets)
                loc_loss = loc_loss_fn(loc_preds, loc_targets)
                loss = cls_loss + loc_loss
                
                val_loss += loss.item()
            
            val_loss /= len(val_dataloader)
            print(f"Validation Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(output_dir, "pointpillars_best.pth"))
                print("Saved best model")
        
        # Save latest model
        torch.save(model.state_dict(), os.path.join(output_dir, "pointpillars_latest.pth"))
    
    print("Training completed!")
    
    # Return best model path
    return os.path.join(output_dir, "pointpillars_best.pth")

def export_to_onnx(model_path, output_dir):
    """Export trained model to ONNX format"""
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PointPillars().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Create dummy inputs
    dummy_voxels = torch.zeros((1, MAX_VOXELS, MAX_POINTS_PER_VOXEL, 4), dtype=torch.float32).to(device)
    dummy_num_points = torch.zeros((1, MAX_VOXELS), dtype=torch.int32).to(device)
    dummy_coords = torch.zeros((1, MAX_VOXELS, 3), dtype=torch.int32).to(device)
    
    # Export PFE
    pfe_path = os.path.join(output_dir, "pfe.onnx")
    torch.onnx.export(
        model.pfe,
        (dummy_voxels, dummy_num_points, dummy_coords),
        pfe_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['voxels', 'num_points', 'coords'],
        output_names=['pillar_features'],
        dynamic_axes={
            'voxels': {0: 'batch_size'},
            'num_points': {0: 'batch_size'},
            'coords': {0: 'batch_size'},
            'pillar_features': {0: 'batch_size'}
        }
    )
    print(f"Exported PFE to {pfe_path}")
    
    # Create intermediate output for RPN
    with torch.no_grad():
        pillar_features = model.pfe(dummy_voxels, dummy_num_points, dummy_coords)
        
        # Scatter to BEV
        batch_size = dummy_voxels.shape[0]
        spatial_features = torch.zeros(
            (batch_size, pillar_features.shape[2], 
             int((POINT_CLOUD_RANGE[3] - POINT_CLOUD_RANGE[0]) / VOXEL_SIZE[0]),
             int((POINT_CLOUD_RANGE[4] - POINT_CLOUD_RANGE[1]) / VOXEL_SIZE[1])),
            device=device
        )
        
        # Simple mock scatter
        indices = torch.randint(
            0, spatial_features.shape[2] * spatial_features.shape[3], 
            (pillar_features.shape[1],)
        )
        x_idx = indices % spatial_features.shape[2]
        y_idx = indices // spatial_features.shape[2]
        spatial_features[0, :, x_idx, y_idx] = pillar_features[0].T
    
    # Export RPN
    rpn_path = os.path.join(output_dir, "rpn.onnx")
    torch.onnx.export(
        model.rpn,
        spatial_features,
        rpn_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['spatial_features'],
        output_names=['cls_preds', 'loc_preds'],
        dynamic_axes={
            'spatial_features': {0: 'batch_size'},
            'cls_preds': {0: 'batch_size'},
            'loc_preds': {0: 'batch_size'}
        }
    )
    print(f"Exported RPN to {rpn_path}")
    
    # Verify exported models
    onnx_model = onnx.load(pfe_path)
    onnx.checker.check_model(onnx_model)
    print("PFE ONNX model checked!")
    
    onnx_model = onnx.load(rpn_path)
    onnx.checker.check_model(onnx_model)
    print("RPN ONNX model checked!")
    
    return pfe_path, rpn_path

def main():
    parser = argparse.ArgumentParser(description="Train PointPillars model on KITTI dataset")
    parser.add_argument("--data_root", type=str, default="data/kitti", 
                       help="Path to KITTI dataset")
    parser.add_argument("--output_dir", type=str, default="models", 
                       help="Directory to save models")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, 
                       help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=EPOCHS, 
                       help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, 
                       help="Learning rate")
    parser.add_argument("--only_export", action="store_true", 
                       help="Skip training and only export model to ONNX")
    parser.add_argument("--model_path", type=str, default=None,
                       help="Path to pretrained model (for export only)")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Select device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if not args.only_export:
        # Create dataset
        train_dataset = KittiDataset(args.data_root, split='train')
        val_dataset = KittiDataset(args.data_root, split='train')  # Use training set for validation
        
        # Create data loaders
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # Create model
        model = PointPillars().to(device)
        
        # Create optimizer and scheduler
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = MultiStepLR(optimizer, milestones=[20, 40, 60], gamma=0.1)
        
        # Train model
        best_model_path = train(
            model, train_dataloader, val_dataloader, 
            optimizer, scheduler, args.epochs, device, args.output_dir
        )
    else:
        best_model_path = args.model_path
        if best_model_path is None:
            best_model_path = os.path.join(args.output_dir, "pointpillars_best.pth")
            if not os.path.exists(best_model_path):
                print(f"Error: Model path {best_model_path} does not exist")
                return
    
    # Export model to ONNX
    pfe_path, rpn_path = export_to_onnx(best_model_path, args.output_dir)
    print(f"Exported models to {pfe_path} and {rpn_path}")

if __name__ == "__main__":
    main()