import sys
import os
import argparse
import cv2
import numpy as np

# Add the parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.face_utils import FaceDetector, FaceRecognizer, build_face_database, capture_face_samples_interactive

def main():
    parser = argparse.ArgumentParser(description="Register faces for recognition system")
    parser.add_argument("--face_detector", type=str, default="models/face_detection_yunet_2023mar.onnx",
                        help="Path to face detector ONNX model")
    parser.add_argument("--face_recognizer", type=str, default="models/face_recognition_sface_2021dec.onnx",
                        help="Path to face recognizer ONNX model")
    parser.add_argument("--faces_dir", type=str, default="data/faces",
                        help="Directory to store face samples")
    parser.add_argument("--db_path", type=str, default="data/db_embeddings.pkl",
                        help="Path to save face database")
    parser.add_argument("--mode", type=str, choices=["capture", "build"], default="build",
                        help="Mode: capture (capture face samples) or build (build database)")
    parser.add_argument("--person_name", type=str, help="Name of person to capture (required for capture mode)")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to capture per person")
    
    args = parser.parse_args()
    
    # Initialize detector and recognizer
    detector = FaceDetector(args.face_detector)
    recognizer = FaceRecognizer(args.face_recognizer)
    
    if args.mode == "capture":
        if not args.person_name:
            print("Error: --person_name is required for capture mode")
            return
        
        # Use the interactive face capture
        print(f"Starting interactive face capture for {args.person_name}...")
        success = capture_face_samples_interactive(detector, args.faces_dir, args.person_name, args.num_samples)
        
        if success:
            print(f"Successfully captured samples for {args.person_name}")
        else:
            print(f"Failed to capture samples for {args.person_name}")
    
    elif args.mode == "build":
        # Build face database from captured samples
        print("Building face database...")
        if not os.path.exists(args.faces_dir):
            print(f"Error: Faces directory {args.faces_dir} does not exist")
            print("Please capture face samples first using --mode capture")
            return
        
        # Build database
        count = build_face_database(detector, recognizer, args.faces_dir, args.db_path)
        print(f"Database built with {count} faces")
        
        # Print summary
        print("\nDatabase summary:")
        recognizer.load_database(args.db_path)
        for name, features in recognizer.database.items():
            print(f"  {name}: {len(features)} faces")

if __name__ == "__main__":
    main()