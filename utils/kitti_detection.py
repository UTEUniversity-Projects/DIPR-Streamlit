import numpy as np
import cv2
import torch
import open3d as o3d
from typing import List, Dict, Tuple, Optional
import os
from pyquaternion import Quaternion

class KITTIDetector:
    def __init__(self, model_path: str):
        """
        Initialize KITTI 3D object detector
        Args:
            model_path: Path to the trained model
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model(model_path)
        self.classes = ['Car', 'Pedestrian', 'Cyclist']
        
    def load_model(self, model_path: str):
        """Load the trained model"""
        # In practice, you would load your specific 3D detection model here
        # For now, we'll create a placeholder
        model = torch.jit.load(model_path, map_location=self.device)
        model.eval()
        return model
    
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
    
    def detect(self, point_cloud: np.ndarray) -> Tuple[List[Dict], List[float]]:
        """
        Detect 3D objects in point cloud
        Args:
            point_cloud: Input point cloud
        Returns:
            detections: List of detected objects
            scores: Confidence scores
        """
        # In practice, this would use your trained model
        # For now, we'll return mock data
        detections = [
            {
                'class': 'Car',
                'location': [5.0, 1.5, 10.0],
                'dimensions': [1.8, 1.6, 4.0],
                'rotation_y': 0.0,
                'bbox': [100, 200, 200, 300]
            }
        ]
        scores = [0.95]
        return detections, scores
    
    def visualize_3d(self, point_cloud: np.ndarray, 
                    detections: List[Dict], 
                    scores: List[float]) -> o3d.geometry.PointCloud:
        """
        Visualize 3D detection results
        Args:
            point_cloud: Input point cloud
            detections: Detected objects
            scores: Detection scores
        Returns:
            vis: 3D visualization object
        """
        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
        
        # Add detection boxes
        vis_objects = [pcd]
        
        for detection in detections:
            # Create bounding box
            location = np.array(detection['location'])
            dimensions = np.array(detection['dimensions'])
            rotation_y = detection['rotation_y']
            
            # Create rotation matrix
            rotation_matrix = np.array([
                [np.cos(rotation_y), 0, np.sin(rotation_y)],
                [0, 1, 0],
                [-np.sin(rotation_y), 0, np.cos(rotation_y)]
            ])
            
            # Create bounding box vertices
            vertices = np.array([
                [-dimensions[0]/2, -dimensions[1]/2, -dimensions[2]/2],
                [dimensions[0]/2, -dimensions[1]/2, -dimensions[2]/2],
                [dimensions[0]/2, dimensions[1]/2, -dimensions[2]/2],
                [-dimensions[0]/2, dimensions[1]/2, -dimensions[2]/2],
                [-dimensions[0]/2, -dimensions[1]/2, dimensions[2]/2],
                [dimensions[0]/2, -dimensions[1]/2, dimensions[2]/2],
                [dimensions[0]/2, dimensions[1]/2, dimensions[2]/2],
                [-dimensions[0]/2, dimensions[1]/2, dimensions[2]/2]
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
            line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in lines])
            
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
        for i, detection in enumerate(detections):
            class_name = detection['class']
            bbox = detection['bbox']
            score = scores[i]
            
            # Draw bounding box
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{class_name}: {score:.2f}"
            cv2.putText(image, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return image

def prepare_kitti_sample(data_dir: str, index: int) -> Dict:
    """Prepare a single KITTI sample"""
    # File paths
    velodyne_path = os.path.join(data_dir, "raw", "data_object_velodyne", 
                                "training", "velodyne", f"{index:06d}.bin")
    image_path = os.path.join(data_dir, "raw", "data_object_image_2", 
                             "training", "image_2", f"{index:06d}.png")
    calib_path = os.path.join(data_dir, "raw", "data_object_calib", 
                             "training", "calib", f"{index:06d}.txt")
    label_path = os.path.join(data_dir, "raw", "data_object_label_2", 
                             "training", "label_2", f"{index:06d}.txt")
    
    # Load data
    detector = KITTIDetector("models/kitti_3d_detection.pt")
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