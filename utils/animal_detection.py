import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Tuple, Optional

class AnimalDetector:
    def __init__(self, model_path: str, conf_threshold: float = 0.25):
        """
        Initialize animal detector with YOLOv8 model
        Args:
            model_path: Path to YOLOv8 model
            conf_threshold: Confidence threshold for detection
        """
        # Load YOLOv8 model
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        
        # Common animal classes that may be in the model
        # These will be overridden by actual classes from the model
        self.class_names = {
            0: "Dog",
            1: "Cat",
            2: "Bird",
            3: "Horse",
            4: "Cow",
            5: "Elephant",
            6: "Bear",
            7: "Zebra",
            8: "Giraffe",
            9: "Tiger"
        }
        
        # Update class names from the model if available
        try:
            model_classes = self.model.names
            if model_classes:
                self.class_names = model_classes
        except:
            pass
    
    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, List[str], List[float]]:
        """
        Detect animals in the input frame
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
        
        # Define color mapping for different animal types
        colors = {
            "Dog": (0, 255, 0),       # Green
            "Cat": (0, 165, 255),     # Orange
            "Bird": (255, 0, 255),    # Magenta
            "Horse": (0, 0, 255),     # Red
            "Cow": (255, 0, 0),       # Blue
            "Elephant": (128, 0, 128), # Purple
            "Bear": (0, 128, 128),    # Teal
            "Zebra": (128, 128, 0),   # Olive
            "Giraffe": (0, 255, 255), # Yellow
            "Tiger": (255, 128, 0)    # Sky Blue
        }
        
        # Draw each detection
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            
            # Get color for this class
            label = labels[i]
            color = colors.get(label, (128, 128, 128))  # Gray default
            
            # Draw bounding box
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
            
            # Draw label and score
            text = f"{label} ({scores[i]:.2f})"
            
            # Calculate text size for better placement
            (text_width, text_height), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            
            # Draw background rectangle for text
            cv2.rectangle(
                result,
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width + 5, y1),
                color,
                -1
            )
            
            # Draw text on background
            cv2.putText(
                result, text, (x1 + 3, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )
        
        return result