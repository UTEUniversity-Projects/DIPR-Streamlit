import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Tuple, Optional

class FruitDetector:
    def __init__(self, model_path: str, conf_threshold: float = 0.25):
        """
        Initialize fruit detector with YOLOv8 model trained on Roboflow dataset
        Args:
            model_path: Path to YOLOv8 model
            conf_threshold: Confidence threshold for detection
        """
        # Load YOLOv8 model
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        
        # Class names from Roboflow dataset (based on data.yaml)
        self.class_names = {
            0: "Apple",
            1: "Banana", 
            2: "Kiwi",
            3: "Orange",
            4: "Pear"
        }
    
    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, List[str], List[float]]:
        """
        Detect fruits in the input frame
        Args:
            frame: Input image frame
        Returns:
            boxes: Bounding boxes in [x1, y1, x2, y2] format
            labels: Class labels
            scores: Confidence scores
        """
        # Run YOLOv8 inference
        results = self.model(frame, verbose=False)[0]
        
        # Extract detection information
        boxes = []
        labels = []
        scores = []
        
        for box in results.boxes:
            # Check confidence threshold
            conf = float(box.conf)
            if conf < self.conf_threshold:
                continue
            
            # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Get class information
            cls_id = int(box.cls)
            cls_name = self.class_names.get(cls_id, f"class_{cls_id}")
            
            # Add to results
            boxes.append([x1, y1, x2, y2])
            labels.append(cls_name)
            scores.append(conf)
        
        return np.array(boxes), labels, scores

    def draw_results(self, frame: np.ndarray, boxes: np.ndarray, 
                    labels: List[str], scores: List[float]) -> np.ndarray:
        """
        Draw detection results on the frame
        Args:
            frame: Input image frame
            boxes: Bounding boxes in [x1, y1, x2, y2] format
            labels: Class labels
            scores: Confidence scores
        Returns:
            frame: Frame with results drawn
        """
        result = frame.copy()
        
        # Define colors for different classes (updated for Roboflow classes)
        colors = {
            "Apple": (0, 255, 0),     # Green
            "Banana": (0, 255, 255),  # Yellow
            "Kiwi": (255, 0, 255),    # Magenta
            "Orange": (0, 127, 255),  # Orange
            "Pear": (255, 0, 0)       # Blue
        }
        
        # Draw each detection
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            
            # Get color for this class
            color = colors.get(labels[i], (128, 128, 128))  # Gray default
            
            # Draw bounding box
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
            
            # Draw label and score
            label = f"{labels[i]} ({scores[i]:.2f})"
            cv2.putText(result, label, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return result