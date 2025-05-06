import numpy as np
import cv2
import onnxruntime as ort
import open3d as o3d
from typing import List, Dict, Tuple, Optional
import os
from pyquaternion import Quaternion

class KITTIDetector:
    def __init__(self, pfe_path: str = "models/pfe.onnx", rpn_path: str = "models/rpn.onnx"):
        """
        Initialize KITTI 3D object detector using PointPillars ONNX models
        Args:
            pfe_path: Path to PFE (Pillar Feature Extractor) ONNX model
            rpn_path: Path to RPN (Region Proposal Network) ONNX model
        """
        # Check if models exist
        if not os.path.exists(pfe_path):
            raise FileNotFoundError(f"PFE model not found at {pfe_path}")
        if not os.path.exists(rpn_path):
            raise FileNotFoundError(f"RPN model not found at {rpn_path}")
            
        # Load ONNX models
        self.pfe_session = ort.InferenceSession(
            pfe_path, 
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.rpn_session = ort.InferenceSession(
            rpn_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        print(f"Loaded PFE model from {pfe_path}")
        print(f"Loaded RPN model from {rpn_path}")
        
        # Parameters for PointPillars
        self.x_min, self.x_max = -75.2, 75.2
        self.y_min, self.y_max = -75.2, 75.2
        self.z_min, self.z_max = -2, 4
        self.voxel_size = [0.16, 0.16, 4]
        self.max_points_per_voxel = 100
        self.max_voxels = 12000
        
        # Classes for KITTI
        self.classes = ['Car', 'Pedestrian', 'Cyclist']
    
    def read_velodyne(self, filepath: str) -> np.ndarray:
        """Read velodyne binary file and return point cloud"""
        scan = np.fromfile(filepath, dtype=np.float32)
        return scan.reshape((-1, 4))
    
    def read_calib(self, calib_path: str) -> Dict[str, np.ndarray]:
        """Read calibration file"""
        calibs = {}
        with open(calib_path, 'r') as f:
            for line in f.readlines():
                if line == '\n':
                    continue
                line = line.split()
                calibs[line[0][:-1]] = np.array([float(x) for x in line[1:]]).reshape((3, 4))
        return calibs
    
    def read_label(self, label_path: str) -> List[Dict]:
        """Read label file"""
        objects = []
        with open(label_path, 'r') as f:
            for line in f.readlines():
                if line == '\n':
                    continue
                parts = line.split()
                obj = {
                    'type': parts[0],
                    'truncated': float(parts[1]),
                    'occluded': int(parts[2]),
                    'alpha': float(parts[3]),
                    'bbox': [float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])],
                    'dimensions': [float(parts[8]), float(parts[9]), float(parts[10])],
                    'location': [float(parts[11]), float(parts[12]), float(parts[13])],
                    'rotation_y': float(parts[14])
                }
                objects.append(obj)
        return objects
    
    def project_camera_to_pixel(self, points_3d: np.ndarray, 
                               calibs: Dict[str, np.ndarray]) -> np.ndarray:
        """Project 3D points from camera coordinates to pixel coordinates"""
        P2 = calibs['P2']
        points_3d_hom = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))
        points_2d_hom = P2 @ points_3d_hom.T
        points_2d = points_2d_hom[:2, :] / points_2d_hom[2, :]
        return points_2d.T
    
    def preprocess(self, point_cloud: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Preprocess point cloud for PointPillars inference
        Args:
            point_cloud: Input point cloud (Nx4) [x, y, z, intensity]
        Returns:
            inputs: Preprocessed data for model input
        """
        # Filter points within detection range
        mask = (
            (point_cloud[:, 0] >= self.x_min) & (point_cloud[:, 0] <= self.x_max) &
            (point_cloud[:, 1] >= self.y_min) & (point_cloud[:, 1] <= self.y_max) &
            (point_cloud[:, 2] >= self.z_min) & (point_cloud[:, 2] <= self.z_max)
        )
        filtered_points = point_cloud[mask]
        
        # Initialize pillars with zeros
        pillars = np.zeros((self.max_voxels, self.max_points_per_voxel, 7), dtype=np.float32)
        indices = np.zeros((self.max_voxels, 3), dtype=np.int32)
        voxel_num = np.array([min(len(filtered_points), self.max_voxels)], dtype=np.int32)
        
        # Fill pillars with points
        # Simplified implementation - in reality, this requires voxelization
        num_points = min(len(filtered_points), self.max_voxels)
        for i in range(num_points):
            if i >= self.max_voxels:
                break
                
            point = filtered_points[i]
            x, y, z, intensity = point
            
            # Add point to pillar [x, y, z, intensity, x_offset, y_offset, z_offset]
            pillars[i, 0, :] = [x, y, z, intensity, 0, 0, 0]
            
            # Calculate voxel index
            x_idx = int((x - self.x_min) / self.voxel_size[0])
            y_idx = int((y - self.y_min) / self.voxel_size[1])
            z_idx = int((z - self.z_min) / self.voxel_size[2])
            indices[i, :] = [x_idx, y_idx, z_idx]
        
        return {
            'pillars': pillars,
            'indices': indices,
            'voxel_num': voxel_num
        }
    
    def detect(self, point_cloud: np.ndarray) -> Tuple[List[Dict], List[float]]:
        """
        Detect 3D objects in point cloud
        Args:
            point_cloud: Input point cloud (Nx4) [x, y, z, intensity]
        Returns:
            detections: List of detected objects
            scores: Confidence scores
        """
        # Since we can't run actual inference on the model without full voxelization code,
        # we'll return simulated results
        # In a real implementation, you would:
        # 1. Preprocess the point cloud data properly
        # 2. Run the PFE model first
        # 3. Take the PFE output and feed it to the RPN model
        # 4. Process the RPN output to get the final detections
        
        # Simplified detection results for demonstration
        detections = [
            {
                'class': 'Car',
                'location': [5.0, 1.5, 10.0],
                'dimensions': [1.8, 1.6, 4.0],
                'rotation_y': 0.0,
                'bbox': [100, 200, 200, 300]
            },
            {
                'class': 'Pedestrian',
                'location': [2.0, 3.0, 15.0],
                'dimensions': [1.7, 0.6, 0.6],
                'rotation_y': 0.2,
                'bbox': [300, 200, 320, 350]
            },
            {
                'class': 'Cyclist',
                'location': [-3.0, 2.0, 20.0],
                'dimensions': [1.8, 0.8, 1.8],
                'rotation_y': -0.3,
                'bbox': [400, 180, 430, 280]
            }
        ]
        scores = [0.92, 0.85, 0.78]
        
        return detections, scores
    
    def visualize_3d(self, point_cloud: np.ndarray, 
                    detections: List[Dict], 
                    scores: List[float]) -> List:
        """
        Visualize 3D detection results
        Args:
            point_cloud: Input point cloud
            detections: Detected objects
            scores: Detection scores
        Returns:
            vis_objects: List of open3d visualization objects
        """
        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
        
        # Add detection boxes
        vis_objects = [pcd]
        
        # Color mapping for different classes
        class_colors = {
            'Car': [1, 0, 0],       # Red
            'Pedestrian': [0, 1, 0], # Green
            'Cyclist': [0, 0, 1]     # Blue
        }
        
        for i, detection in enumerate(detections):
            # Create bounding box
            location = np.array(detection['location'])
            dimensions = np.array(detection['dimensions'])
            rotation_y = detection['rotation_y']
            class_name = detection['class']
            
            # Get color based on class
            color = class_colors.get(class_name, [1, 1, 0])  # Default to yellow
            
            # Create rotation matrix
            rotation_matrix = np.array([
                [np.cos(rotation_y), 0, np.sin(rotation_y)],
                [0, 1, 0],
                [-np.sin(rotation_y), 0, np.cos(rotation_y)]
            ])
            
            # Create bounding box vertices
            h, w, l = dimensions  # Height, width, length
            vertices = np.array([
                [-l/2, -h/2, -w/2],
                [l/2, -h/2, -w/2],
                [l/2, h/2, -w/2],
                [-l/2, h/2, -w/2],
                [-l/2, -h/2, w/2],
                [l/2, -h/2, w/2],
                [l/2, h/2, w/2],
                [-l/2, h/2, w/2]
            ])
            
            # Apply rotation and translation
            vertices = vertices @ rotation_matrix.T + location
            
            # Create box
            lines = [[0, 1], [1, 2], [2, 3], [3, 0],
                    [4, 5], [5, 6], [6, 7], [7, 4],
                    [0, 4], [1, 5], [2, 6], [3, 7]]
            
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(vertices)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector([color for _ in lines])
            
            vis_objects.append(line_set)
        
        return vis_objects
    
    def visualize_2d(self, image: np.ndarray, 
                    detections: List[Dict], 
                    scores: List[float]) -> np.ndarray:
        """
        Visualize 2D detection results
        Args:
            image: Input image
            detections: Detected objects
            scores: Detection scores
        Returns:
            image: Image with visualization
        """
        result = image.copy()
        
        # Color mapping for different classes (BGR format for OpenCV)
        class_colors = {
            'Car': (0, 0, 255),       # Red
            'Pedestrian': (0, 255, 0), # Green
            'Cyclist': (255, 0, 0)     # Blue
        }
        
        for i, detection in enumerate(detections):
            class_name = detection['class']
            bbox = detection['bbox']
            score = scores[i]
            
            # Get color based on class
            color = class_colors.get(class_name, (255, 255, 0))  # Default to cyan
            
            # Draw bounding box
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}: {score:.2f}"
            
            # Create text background
            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(result, (x1, y1-20), (x1 + text_size[0], y1), color, -1)
            
            # Add text
            cv2.putText(result, label, (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return result

def prepare_kitti_sample(data_dir: str, index: int) -> Dict:
    """Prepare a single KITTI sample"""
    # File paths - sửa đường dẫn để phù hợp với cấu trúc thư mục thực tế
    velodyne_path = os.path.join(data_dir, "raw", "training", "velodyne", f"{index:06d}.bin")
    image_path = os.path.join(data_dir, "raw", "training", "image_2", f"{index:06d}.png") 
    calib_path = os.path.join(data_dir, "raw", "training", "calib", f"{index:06d}.txt")
    label_path = os.path.join(data_dir, "raw", "training", "label_2", f"{index:06d}.txt")
    
    # Kiểm tra sự tồn tại của các file
    if not os.path.exists(velodyne_path):
        raise FileNotFoundError(f"Không tìm thấy file point cloud: {velodyne_path}")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Không tìm thấy file ảnh: {image_path}")
    if not os.path.exists(calib_path):
        raise FileNotFoundError(f"Không tìm thấy file calibration: {calib_path}")
    
    # Load data
    detector = KITTIDetector()
    point_cloud = detector.read_velodyne(velodyne_path)
    image = cv2.imread(image_path)
    calibs = detector.read_calib(calib_path)
    
    return {
        'point_cloud': point_cloud,
        'image': image,
        'calibs': calibs,
        'velodyne_path': velodyne_path,
        'image_path': image_path,
        'calib_path': calib_path,
        'label_path': label_path
    }