import cv2
import numpy as np
import os
import pickle
from sklearn.preprocessing import normalize
from datetime import datetime
from typing import List, Dict, Tuple, Optional

class FaceDetector:
    def __init__(self, model_path: str, score_threshold: float = 0.9, nms_threshold: float = 0.6):
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
    
    def setInputSize(self, input_size: Tuple[int, int]):
        """Set input size for the detector"""
        self.model.setInputSize(input_size)
        
    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Detect faces in the input frame
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
        
        # Process detected faces
        aligned_faces = []
        for face in faces:
            # Extract face box coordinates and convert to int
            x, y, w, h = map(int, face[:4])
            
            # Add padding to make the crop more square
            # Calculate square size
            size = max(w, h)
            padding = int(size * 0.1)  # 10% padding
            size += padding * 2
            
            # Calculate center and new coordinates
            center_x = x + w // 2
            center_y = y + h // 2
            
            x_new = center_x - size // 2
            y_new = center_y - size // 2
            
            # Make sure coordinates are within bounds
            x_new = max(0, x_new)
            y_new = max(0, y_new)
            x2_new = min(x_new + size, frame.shape[1])
            y2_new = min(y_new + size, frame.shape[0])
            
            # Ensure we have a valid face crop
            if x2_new > x_new and y2_new > y_new:
                aligned_face = frame[y_new:y2_new, x_new:x2_new]
                
                # Check if the cropped face is valid and resize
                if aligned_face.size > 0:
                    # Resize to 112x112 for SFace model
                    aligned_face = cv2.resize(aligned_face, (112, 112))
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
    
    def get_feature(self, face_img: np.ndarray) -> np.ndarray:
        """
        Extract face feature from aligned face image
        Args:
            face_img: Aligned face image
        Returns:
            feature: Face feature embedding vector
        """
        # Make sure the face image is properly sized for the model
        face_img = cv2.resize(face_img, (112, 112))
        # Extract face feature
        feature = self.model.feature(face_img)
        # Normalize feature vector
        feature = normalize(feature.reshape(1, -1))[0]
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
    
    def identify(self, face_img: np.ndarray, threshold: float = 0.6) -> Tuple[str, float]:
        """
        Identify person from face image
        Args:
            face_img: Aligned face image
            threshold: Recognition threshold
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
        
    def identify_with_flip(self, face_img: np.ndarray, threshold: float = 0.6) -> Tuple[str, float, bool]:
        """
        Identify person from face image, trying both original and flipped versions
        Args:
            face_img: Aligned face image
            threshold: Recognition threshold
        Returns:
            name: Person name or "Unknown"
            score: Confidence score
            flipped: True if recognized from flipped image
        """
        # Try original image
        name1, score1 = self.identify(face_img, threshold)
        
        # If recognized, return immediately
        if name1 != "Unknown":
            return name1, score1, False
        
        # Try flipped image
        face_img_flipped = cv2.flip(face_img, 1)  # Flip horizontally
        name2, score2 = self.identify(face_img_flipped, threshold)
        
        # Return the better result
        if score2 > score1:
            return name2, score2, True
        else:
            return name1, score1, False
    
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

def capture_face_samples(detector: FaceDetector, output_dir: str, 
                         person_name: str, num_samples: int = 100, 
                         delay_frames: int = 3) -> bool:
    """
    Capture face samples from webcam
    Args:
        detector: Face detector object
        output_dir: Output directory for face images
        person_name: Name of the person
        num_samples: Number of samples to capture
        delay_frames: Number of frames to skip between captures
    Returns:
        success: Whether capture was successful
    """
    # Create person directory if it doesn't exist
    person_dir = os.path.join(output_dir, person_name)
    os.makedirs(person_dir, exist_ok=True)
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return False
    
    # Counter variables
    count = 0
    frame_count = 0
    
    print(f"Starting capture for {person_name}. Press 'q' to quit.")
    
    while count < num_samples:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break
        
        # Flip the frame horizontally for a more natural view
        frame = cv2.flip(frame, 1)
        
        # Detect faces
        faces, aligned_faces = detector.detect(frame)
        
        # Draw detected faces
        if len(faces) > 0:
            for face in faces:
                x, y, w, h = map(int, face[:4])
                confidence = face[4]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"Confidence: {confidence:.2f}", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Display instructions and count
        cv2.putText(frame, f"Captured: {count}/{num_samples}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "Press 'q' to quit", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display the frame
        cv2.imshow("Face Capture", frame)
        
        # Process key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        
        # Save face if detected
        frame_count += 1
        if frame_count % delay_frames == 0 and len(aligned_faces) > 0:
            # Save the first face detected
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = os.path.join(person_dir, f"{count:04d}_{timestamp}.jpg")
            cv2.imwrite(filename, aligned_faces[0])
            count += 1
            print(f"Saved {count}/{num_samples}")
    
    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"Captured {count} samples for {person_name}")
    return count > 0

def capture_face_samples_interactive(detector: FaceDetector, output_dir: str, 
                                   person_name: str, num_samples: int = 100) -> bool:
    """
    Capture face samples with keyboard controls:
    - Press 's' to start/resume capturing
    - Press 'p' to pause capturing  
    - Press 'q' to quit
    - Continues until 100 samples or manually stopped
    """
    # Create person directory if it doesn't exist
    person_dir = os.path.join(output_dir, person_name)
    os.makedirs(person_dir, exist_ok=True)
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return False
    
    # State variables
    count = 0
    capturing = False
    frame_count = 0
    delay_frames = 3
    
    # Set camera resolution for consistent UI layout
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print(f"Interactive face capture for {person_name}")
    print("Controls:")
    print("  's' - Start/Resume capturing")
    print("  'p' - Pause capturing") 
    print("  'q' - Quit")
    print("Tips:")
    print("  - Keep face centered and well-lit")
    print("  - Turn head slightly for different angles")
    
    while count < num_samples:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break
        
        # Flip the frame horizontally
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        
        # Create capture zone (larger area) with better positioning
        zone_width = min(int(w * 0.5), 640)  # Max 640px wide
        zone_height = min(int(h * 0.5), 480)  # Max 480px high
        zone_x = max(0, int(w * 0.25))  # Centered
        zone_y = max(0, int(h * 0.25))  # Centered
        
        # Create a region for text overlays
        text_region = frame.copy()
        
        # Draw capture zone on the main frame
        cv2.rectangle(frame, (zone_x, zone_y), 
                     (zone_x + zone_width, zone_y + zone_height), 
                     (0, 255, 0), 3)
        
        # Detect faces
        faces, _ = detector.detect(frame)
        
        best_face = None
        face_to_save = None
        
        # Process detected faces
        if len(faces) > 0:
            for face in faces:
                x, y, w_face, h_face = map(int, face[:4])
                
                # Draw face rectangle in blue (for display only)
                cv2.rectangle(frame, (x, y), (x+w_face, y+h_face), (255, 0, 0), 2)
                
                # Draw face center
                face_center_x = x + w_face // 2
                face_center_y = y + h_face // 2
                cv2.circle(frame, (face_center_x, face_center_y), 5, (255, 0, 0), -1)
                
                # Select best face (largest)
                if best_face is None or w_face * h_face > best_face[2] * best_face[3]:
                    best_face = (x, y, w_face, h_face)
                    
                    # Crop face area from ORIGINAL frame (without rectangles) for saving
                    padding = int(max(w_face, h_face) * 0.2)  # 20% padding
                    x1 = max(0, x - padding)
                    y1 = max(0, y - padding)
                    x2 = min(w, x + w_face + padding)
                    y2 = min(h, y + h_face + padding)
                    
                    # Get clean frame without rectangles
                    ret, clean_frame = cap.read()
                    clean_frame = cv2.flip(clean_frame, 1)
                    
                    face_crop = clean_frame[y1:y2, x1:x2].copy()
                    
                    # Resize to standard size
                    face_to_save = cv2.resize(face_crop, (112, 112))
        
        # Create text background to prevent overlap
        text_bg = np.zeros((150, w, 3), dtype=np.uint8)
        text_bg[:, :] = (50, 50, 50)  # Dark gray background
        
        # Status text
        status = "CAPTURING" if capturing else "PAUSED"
        color = (0, 255, 0) if capturing else (0, 0, 255)
        
        # Add text to background
        cv2.putText(text_bg, f"Status: {status}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        cv2.putText(text_bg, f"Count: {count}/{num_samples}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Face detection status
        if len(faces) > 0:
            cv2.putText(text_bg, f"Faces detected: {len(faces)}", (20, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(text_bg, "No face detected", (20, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Add text background to frame
        combined = frame.copy()
        combined[0:150, :] = text_bg
        
        # Control instructions at the bottom (with background)
        bottom_text_bg = np.zeros((40, w, 3), dtype=np.uint8)
        bottom_text_bg[:, :] = (50, 50, 50)  # Dark gray background
        
        cv2.putText(bottom_text_bg, "Press 's' to start, 'p' to pause, 'q' to quit", (20, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add bottom text to frame
        combined[h-40:h, :] = bottom_text_bg
        
        # Show the frame
        cv2.imshow(f"Face Capture - {person_name}", combined)
        
        # Process keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('s'):
            capturing = True
            print(f"Started capturing... ({count}/{num_samples})")
        elif key == ord('p'):
            capturing = False
            print(f"Paused capturing ({count}/{num_samples})")
        elif key == ord('q'):
            print(f"Quitting... Captured {count} images")
            break
        
        # Save face if capturing is active and we have a face
        if capturing and face_to_save is not None:
            frame_count += 1
            if frame_count % delay_frames == 0:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = os.path.join(person_dir, f"{count:04d}_{timestamp}.jpg")
                cv2.imwrite(filename, face_to_save)
                count += 1
                print(f"Captured {count}/{num_samples} images")
                
                # Don't auto-pause, let user control manually
                # Notify every 10 images
                if count % 10 == 0 and count < num_samples:
                    print(f"Progress: {count} images captured. Press 'p' to pause, 's' to continue, 'q' to quit.")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\nCompleted! Captured {count} samples for {person_name}")
    return count > 0

def enhance_face_image(face_img: np.ndarray) -> np.ndarray:
    """
    Enhance face image for better recognition
    """
    # Resize to standard size
    face_img = cv2.resize(face_img, (112, 112))
    
    # Apply histogram equalization
    if len(face_img.shape) == 2:  # Grayscale
        face_img = cv2.equalizeHist(face_img)
    else:  # Color
        # Convert to LAB and equalize L channel
        lab = cv2.cvtColor(face_img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.equalizeHist(l)
        lab = cv2.merge([l, a, b])
        face_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # Apply slight blur to reduce noise
    face_img = cv2.GaussianBlur(face_img, (3, 3), 0)
    
    return face_img

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
            
            # Skip non-image files
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
        x, y, w, h = map(int, face[:4])
        
        # Get name
        name = names[i] if i < len(names) else "Unknown"
        score = scores[i] if i < len(scores) else 0.0
        
        # Choose color based on recognition status
        if name == "Unknown":
            color = (0, 0, 255)  # Red for unknown
            text_color = (0, 0, 255)
        else:
            color = (0, 255, 0)  # Green for known
            text_color = (0, 255, 0)
        
        # Draw face rectangle
        cv2.rectangle(result, (x, y), (x+w, y+h), color, 2)
        
        # Prepare text to display
        label = f"{name} ({score:.2f})"
        
        # Calculate text size and position
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        
        # Create background rectangle for text
        text_y = y - 10 if y - 10 > 0 else y + h + 20
        
        # Draw text background
        cv2.rectangle(result, 
                     (x, text_y - text_height - baseline), 
                     (x + text_width + 5, text_y + baseline),
                     color, 
                     cv2.FILLED)
        
        # Draw text (name and score)
        cv2.putText(result, label, (x + 2, text_y - 5),
                    font, font_scale, (255, 255, 255), thickness)
    
    return result
