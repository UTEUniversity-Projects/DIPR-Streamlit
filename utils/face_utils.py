import cv2
import numpy as np
import os
import pickle
from sklearn.preprocessing import normalize
from datetime import datetime
from typing import List, Dict, Tuple, Optional

class FaceDetector:
    def __init__(self, model_path: str, score_threshold: float = 0.9, nms_threshold: float = 0.3):
        """
        Initialize YuNet face detector
        Args:
            model_path: Path to YuNet .onnx model
            score_threshold: Detection score threshold
            nms_threshold: Non-Maximum Suppression threshold
        """
        self.model = cv2.FaceDetectorYN.create(
            model_path,
            "",
            (320, 320),
            score_threshold,
            nms_threshold,
            5000  # Top K
        )
        # Caching frame size to avoid unnecessary setInputSize calls
        self._last_frame_size = None
    
    def setInputSize(self, input_size: Tuple[int, int]):
        """Set input size for the detector only if it changed"""
        if self._last_frame_size != input_size:
            self.model.setInputSize(input_size)
            self._last_frame_size = input_size
        
    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Detect faces in the input frame - optimized version
        Args:
            frame: Input image frame
        Returns:
            faces: Detected faces as array of [x, y, w, h, score, landmarks[12]]
            aligned_faces: List of aligned face images for recognition
        """
        # Set input size based on frame size
        h, w, _ = frame.shape
        self.setInputSize((w, h))
        
        # Detect faces
        faces = self.model.detect(frame)[1]
        
        # If no faces detected, return empty results
        if faces is None:
            return np.array([]), []
        
        # Process detected faces - optimized with vectorization
        aligned_faces = []
        
        # Thêm padding cho tất cả khuôn mặt cùng lúc
        padding_ratio = 0.1  # 10% padding
        x_coords = faces[:, 0].astype(int)
        y_coords = faces[:, 1].astype(int)
        widths = faces[:, 2].astype(int)
        heights = faces[:, 3].astype(int)
        
        # Tính toán kích thước cân bằng
        sizes = np.maximum(widths, heights)
        paddings = (sizes * padding_ratio).astype(int)
        sizes += paddings * 2
        
        # Tính tọa độ tâm
        center_x = x_coords + widths // 2
        center_y = y_coords + heights // 2
        
        # Tính tọa độ mới với padding
        x_new = np.maximum(0, center_x - sizes // 2)
        y_new = np.maximum(0, center_y - sizes // 2)
        x2_new = np.minimum(x_new + sizes, w)
        y2_new = np.minimum(y_new + sizes, h)
        
        # Crop và resize cho mỗi khuôn mặt
        for i in range(len(faces)):
            # Đảm bảo kích thước crop hợp lệ
            if x2_new[i] > x_new[i] and y2_new[i] > y_new[i]:
                aligned_face = frame[y_new[i]:y2_new[i], x_new[i]:x2_new[i]]
                
                # Kiểm tra nếu crop thành công
                if aligned_face.size > 0:
                    # Resize hiệu quả với interpolation=cv2.INTER_AREA cho tốc độ tốt hơn
                    aligned_face = cv2.resize(aligned_face, (112, 112), interpolation=cv2.INTER_AREA)
                    aligned_faces.append(aligned_face)
        
        return faces, aligned_faces

class FaceRecognizer:
    def __init__(self, model_path: str, database_path: str = None):
        """
        Initialize SFace face recognizer
        Args:
            model_path: Path to SFace .onnx model
            database_path: Optional path to face database file
        """
        self.model = cv2.FaceRecognizerSF.create(
            model_path,
            ""
        )
        
        self.database = {}
        if database_path and os.path.exists(database_path):
            self.load_database(database_path)
            
        # Cache cho vectors đã trích xuất để tránh tính toán lặp lại
        self.feature_cache = {}
    
    def get_feature(self, face_img: np.ndarray) -> np.ndarray:
        """
        Extract face feature from aligned face image with caching
        Args:
            face_img: Aligned face image
        Returns:
            feature: Face feature embedding vector
        """
        # Create a hash key for this image
        img_hash = hash(face_img.tobytes())
        
        # Return cached feature if available
        if img_hash in self.feature_cache:
            return self.feature_cache[img_hash]
            
        # Make sure the face image is properly sized for the model
        if face_img.shape[:2] != (112, 112):
            face_img = cv2.resize(face_img, (112, 112), interpolation=cv2.INTER_AREA)
            
        # Extract face feature
        feature = self.model.feature(face_img)
        # Normalize feature vector
        feature = normalize(feature.reshape(1, -1))[0]
        
        # Cache the result
        self.feature_cache[img_hash] = feature
        
        # Limit cache size to prevent memory leaks
        if len(self.feature_cache) > 1000:
            # Remove a random item when cache gets too large
            self.feature_cache.pop(next(iter(self.feature_cache)))
            
        return feature
    
    def match(self, feature1: np.ndarray, feature2: np.ndarray) -> float:
        """
        Match two face features
        Args:
            feature1: First face feature
            feature2: Second face feature
        Returns:
            score: Similarity score
        """
        # Calculate cosine similarity
        return self.model.match(feature1, feature2, cv2.FaceRecognizerSF_FR_COSINE)
    
    def add_face(self, name: str, face_img: np.ndarray) -> bool:
        """
        Add face to the database
        Args:
            name: Person name
            face_img: Aligned face image
        Returns:
            success: Whether addition was successful
        """
        try:
            feature = self.get_feature(face_img)
            
            if name not in self.database:
                self.database[name] = []
            
            self.database[name].append(feature)
            return True
        except Exception as e:
            print(f"Error adding face: {e}")
            return False
    
    def identify(self, face_img: np.ndarray, threshold: float = 0.45) -> Tuple[str, float]:
        """
        Identify person from face image
        Args:
            face_img: Aligned face image
            threshold: Recognition threshold (lowered to 0.45)
        Returns:
            name: Person name or "Unknown"
            score: Confidence score
        """
        try:
            # Get face feature
            feature = self.get_feature(face_img)
            
            best_score = 0
            best_name = "Unknown"
            
            # Compare with all faces in database
            for name, features in self.database.items():
                for ref_feature in features:
                    score = self.match(feature, ref_feature)
                    if score > best_score:
                        best_score = score
                        best_name = name
            
            # Return Unknown if score is below threshold
            if best_score < threshold:
                return "Unknown", best_score
            
            return best_name, best_score
        
        except Exception as e:
            print(f"Error in face identification: {e}")
            return "Error", 0.0
    
    def save_database(self, path: str) -> bool:
        """
        Save face database to file
        Args:
            path: File path
        Returns:
            success: Whether save was successful
        """
        try:
            with open(path, 'wb') as f:
                pickle.dump(self.database, f)
            return True
        except Exception as e:
            print(f"Error saving database: {e}")
            return False
    
    def load_database(self, path: str) -> bool:
        """
        Load face database from file
        Args:
            path: File path
        Returns:
            success: Whether load was successful
        """
        try:
            with open(path, 'rb') as f:
                self.database = pickle.load(f)
            return True
        except Exception as e:
            print(f"Error loading database: {e}")
            return False

def build_face_database(detector: FaceDetector, recognizer: FaceRecognizer, 
                       faces_dir: str, database_path: str) -> int:
    """
    Build face database from directory of face images
    Args:
        detector: Face detector object
        recognizer: Face recognizer object
        faces_dir: Directory containing subdirectories of face images
        database_path: Output path for database file
    Returns:
        count: Number of faces added to database
    """
    count = 0
    
    # Iterate through person directories
    for person_name in os.listdir(faces_dir):
        person_dir = os.path.join(faces_dir, person_name)
        
        # Skip non-directories
        if not os.path.isdir(person_dir):
            continue
        
        print(f"Processing {person_name}...")
        
        # Process each image in the person directory
        for img_file in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_file)
            
            # Skip non-image files - UPDATED TO INCLUDE BMP
            if not img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                continue
            
            # Read image
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not read {img_path}")
                continue
            
            # Detect faces
            faces, aligned_faces = detector.detect(img)
            
            # Add face to database if detected
            if len(aligned_faces) > 0:
                if recognizer.add_face(person_name, aligned_faces[0]):
                    count += 1
                    print(f"Added face {count} for {person_name}")
    
    # Save the database
    if count > 0:
        print(f"Saving database with {count} faces to {database_path}")
        recognizer.save_database(database_path)
    
    return count

def draw_results(frame: np.ndarray, faces: np.ndarray, names: List[str], 
                scores: List[float]) -> np.ndarray:
    """
    Draw detection and recognition results on the frame
    Args:
        frame: Input image frame
        faces: Detected faces
        names: Recognized names
        scores: Recognition scores
    Returns:
        frame: Frame with results drawn
    """
    result = frame.copy()
    
    # Draw each face with name and score
    for i, face in enumerate(faces):
        # Skip faces with low confidence scores
        if i >= len(scores) or scores[i] < 0.45:
            continue
            
        x, y, w, h = map(int, face[:4])
        
        # Get name
        name = names[i] if i < len(names) else "Unknown"
        score = scores[i] if i < len(scores) else 0.0
        
        # Choose color based on recognition status
        if name == "Unknown":
            color = (0, 0, 255)  # Red for unknown
            text_color = (255, 255, 255)
        else:
            color = (0, 255, 0)  # Green for known
            text_color = (255, 255, 255)
        
        # Xác định độ dày đường viền khung và cỡ chữ nhất quán
        box_thickness = 2
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6  # Cố định font size để nhất quán
        text_thickness = 2  # Cố định độ dày văn bản
        
        # Draw face rectangle với độ dày cố định
        cv2.rectangle(result, (x, y), (x+w, y+h), color, box_thickness)
        
        # Prepare text to display
        label = f"{name} ({score:.2f})"
        
        # Calculate text size với thông số cố định
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)
        
        # Tính toán vị trí text - đặt phía trên khung
        margin = 5  # Margin giữa text và khung
        text_x = x
        text_y = max(y - margin, text_height + baseline)  # Đảm bảo text không bị cắt
        
        # Draw text background tách biệt với khung
        cv2.rectangle(result, 
                     (text_x, text_y - text_height - baseline - margin),
                     (text_x + text_width + margin, text_y),
                     color, 
                     cv2.FILLED)
        
        # Draw text với độ dày cố định
        cv2.putText(result, 
                    label, 
                    (text_x + margin//2, text_y - margin),
                    font, 
                    font_scale, 
                    text_color, 
                    text_thickness)
    
    return result

def capture_face_samples_interactive(output_dir: str, person_name: str, num_samples: int = 100):
    """
    Capture face samples interactively using webcam
    Args:
        output_dir: Directory to save face samples
        person_name: Name of the person to capture
        num_samples: Number of samples to capture
    """
    import cv2
    import os
    from datetime import datetime
    
    # Create face detector
    detector = FaceDetector("models/face_detection_yunet_2023mar.onnx")
    
    # Create output directory if not exists
    person_dir = os.path.join(output_dir, person_name)
    os.makedirs(person_dir, exist_ok=True)
    
    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return
    
    # Get camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Initialize variables
    count = 0
    last_capture_time = datetime.now()
    min_capture_interval = 0.2  # Minimum interval between captures in seconds
    
    print(f"Capturing {num_samples} samples for {person_name}...")
    print("Press 'q' to quit, 'p' to pause/resume")
    
    paused = False
    
    while count < num_samples:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot read frame")
            break
        
        # Mirror image
        frame = cv2.flip(frame, 1)
        
        # Detect faces
        faces, aligned_faces = detector.detect(frame)
        
        # Draw faces on frame
        for face in faces:
            x, y, w, h = map(int, face[:4])
            confidence = face[4]
            
            # Draw face rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"Confidence: {confidence:.2f}", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw status
        status = "PAUSED" if paused else "CAPTURING"
        cv2.putText(frame, f"Status: {status}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if paused else (0, 255, 0), 2)
        cv2.putText(frame, f"Captured: {count}/{num_samples}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display frame
        cv2.imshow(f"Capturing faces for {person_name}", frame)
        
        # Capture face if not paused
        if not paused and len(aligned_faces) > 0:
            current_time = datetime.now()
            time_diff = (current_time - last_capture_time).total_seconds()
            
            if time_diff >= min_capture_interval:
                # Save the first face detected
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = os.path.join(person_dir, f"{count:04d}_{timestamp}.jpg")
                cv2.imwrite(filename, aligned_faces[0])
                
                # Update count and time
                count += 1
                last_capture_time = current_time
                
                print(f"Captured {count}/{num_samples}")
        
        # Wait for key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
            print(f"{'Paused' if paused else 'Resumed'} capturing")
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"Captured {count}/{num_samples} samples for {person_name}")
    return count