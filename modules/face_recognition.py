import streamlit as st
import cv2
import numpy as np
import time
import os
from typing import List, Dict, Tuple, Optional

@st.cache_resource
def load_face_detector():
    from utils.face_utils import FaceDetector
    return FaceDetector("models/face_detection_yunet_2023mar.onnx")

@st.cache_resource
def load_face_recognizer():
    from utils.face_utils import FaceRecognizer
    return FaceRecognizer("models/face_recognition_sface_2021dec.onnx", "data/db_embeddings.pkl")

class StableFaceTracker:
    def __init__(self, max_distance: float = 50.0, min_frames: int = 5, max_missing_frames: int = 15):
        self.tracked_faces: List[Dict] = []
        self.max_distance = max_distance
        self.min_frames = min_frames
        self.max_missing_frames = max_missing_frames
        
    def update(self, detected_faces: np.ndarray, frame_num: int) -> List[Dict]:
        """Update tracked faces with new detections"""
        # If no detections, gradually remove tracked faces
        if len(detected_faces) == 0:
            # Remove faces that have been missing for too long
            self.tracked_faces = [face for face in self.tracked_faces 
                                if frame_num - face['last_seen'] < self.max_missing_frames]
            
            # Gradually decrease confidence for missing faces
            for face in self.tracked_faces:
                face['score'] *= 0.95  # Decay score each frame
                face['frame_count'] = max(0, face['frame_count'] - 1)
                if face['frame_count'] < self.min_frames:
                    face['is_stable'] = False
            
            return self.tracked_faces
        
        # Track existing faces first
        if len(self.tracked_faces) > 0:
            # Mark all as unmatched initially
            unmatched_tracked = list(range(len(self.tracked_faces)))
            matched_new = set()
            
            # Find matches between new and tracked faces
            for j, new_face in enumerate(detected_faces):
                best_match_idx = None
                min_dist = float('inf')
                
                for i in unmatched_tracked:
                    tracked = self.tracked_faces[i]
                    dist = self._calculate_distance(new_face[:4], tracked['box'])
                    if dist < min_dist and dist < self.max_distance:
                        min_dist = dist
                        best_match_idx = i
                
                if best_match_idx is not None:
                    # Update existing face
                    tracked = self.tracked_faces[best_match_idx]
                    tracked['box'] = self._smooth_box(tracked['box'], new_face[:4])
                    tracked['score'] = 0.9 * tracked['score'] + 0.1 * new_face[4]
                    tracked['frame_count'] += 1
                    tracked['last_seen'] = frame_num
                    
                    # Mark as stable if seen for enough frames
                    if tracked['frame_count'] >= self.min_frames:
                        tracked['is_stable'] = True
                    
                    unmatched_tracked.remove(best_match_idx)
                    matched_new.add(j)
            
            # Add new faces
            for j, new_face in enumerate(detected_faces):
                if j not in matched_new:
                    self.tracked_faces.append({
                        'box': new_face[:4],
                        'score': new_face[4],
                        'frame_count': 1,
                        'last_seen': frame_num,
                        'is_stable': False
                    })
        else:
            # Add all detected faces as new
            for new_face in detected_faces:
                self.tracked_faces.append({
                    'box': new_face[:4],
                    'score': new_face[4],
                    'frame_count': 1,
                    'last_seen': frame_num,
                    'is_stable': False
                })
        
        # Remove old faces
        self.tracked_faces = [face for face in self.tracked_faces 
                            if frame_num - face['last_seen'] < self.max_missing_frames]
        
        return self.tracked_faces
    
    def _calculate_distance(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Calculate distance between two face boxes"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        center1 = (x1 + w1/2, y1 + h1/2)
        center2 = (x2 + w2/2, y2 + h2/2)
        return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    def _smooth_box(self, old_box: np.ndarray, new_box: np.ndarray, alpha: float = 0.3) -> np.ndarray:
        """Smooth transition between box positions"""
        return old_box * (1 - alpha) + new_box * alpha

def draw_stable_results(frame: np.ndarray, stable_faces: List[Dict], 
                       names: List[str] = None, scores: List[float] = None) -> np.ndarray:
    """Draw detection and recognition results on the frame with stability (OPTIMIZED)"""
    result = frame.copy()
    
    # Pre-calculate font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    
    # Draw only stable faces
    stable_index = 0
    for face in stable_faces:
        if not face['is_stable']:
            continue
            
        x, y, w, h = map(int, face['box'])
        
        # Get name and score if available
        name = names[stable_index] if names and stable_index < len(names) else None
        score = scores[stable_index] if scores and stable_index < len(scores) else None
        
        # Choose color based on recognition status
        color = (0, 255, 0)  # Green for both detected and recognized faces
        text_color = (255, 255, 255)  # White text
        
        # Draw face rectangle (simpler draw method)
        cv2.rectangle(result, (x, y), (x+w, y+h), color, 2)
        
        # Draw label if available
        if name is not None:
            label = f"{name} ({score:.2f})" if score is not None else name
            
            # Get text size once
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            # Calculate text position
            text_y = y - 10 if y - 10 > 10 else y + h + 20
            
            # Draw text background (simpler rectangle)
            cv2.rectangle(result, 
                         (x, text_y - text_height - baseline), 
                         (x + text_width + 5, text_y + baseline),
                         color, cv2.FILLED)
            
            # Draw text
            cv2.putText(result, label, (x + 2, text_y - 5),
                        font, font_scale, text_color, thickness)
        
        stable_index += 1
    
    return result

def show():
    # Thêm phần giới thiệu và hướng dẫn
    with st.expander("🔍 Giới thiệu về nhận dạng khuôn mặt", expanded=False):
        st.markdown("""
        ### Giới thiệu về nhận dạng khuôn mặt
        
        Tính năng nhận dạng khuôn mặt trong ứng dụng này sử dụng kiến trúc hai giai đoạn hiện đại:
        
        1. **YuNet Face Detector**: Mô hình phát hiện khuôn mặt dựa trên CNN (Convolutional Neural Network) được tối ưu hóa
           - Kiến trúc: Lightweight network với các khối ResNet cải tiến
           - Hiệu suất: Độ chính xác cao và xử lý thời gian thực trên CPU
           - Định dạng: ONNX với kích thước nhỏ (< 1MB)
        
        2. **SFace Recognition**: Mô hình nhận dạng khuôn mặt với độ chính xác cao
           - Kiến trúc: Dựa trên mạng ResNet-50 cải tiến cho các embedding vector 128 chiều
           - Training: Được huấn luyện trên dataset lớn với loss function đặc biệt giúp phân biệt tốt các khuôn mặt
           - Phương pháp: So sánh vector embedding với các vector trong cơ sở dữ liệu
        
        Ứng dụng cũng tích hợp thuật toán theo dõi khuôn mặt (face tracking) để đảm bảo ổn định khi nhận dạng video thời gian thực, giảm hiện tượng "chớp nháy" khi xác định danh tính.
        
        **Các tính năng chính:**
        - Nhận dạng khuôn mặt từ ảnh tĩnh (tải lên hoặc chụp từ webcam)
        - Nhận dạng khuôn mặt từ video tải lên (nhiều định dạng như MP4, AVI, MOV, v.v.)
        - Phát hiện và nhận dạng khuôn mặt từ webcam theo thời gian thực
        - Theo dõi khuôn mặt ổn định giúp giảm đáng kể sai số trong nhận dạng
        
        **Ứng dụng thực tế:**
        - Hệ thống bảo mật và kiểm soát truy cập
        - Hệ thống điểm danh tự động
        - Phân tích video giám sát
        - Nhận dạng khuôn mặt trong dữ liệu hình ảnh và video lớn
        - Trải nghiệm cá nhân hóa trong các hệ thống thông minh
        - Xác thực danh tính không tiếp xúc
        """)
            
    with st.expander("📋 Hướng dẫn sử dụng", expanded=False):
        st.markdown("""
        ### Hướng dẫn sử dụng
        
        #### 1. Chế độ ảnh tĩnh
        - **Upload ảnh**: Tải lên ảnh chứa khuôn mặt cần nhận dạng
        - **Chụp từ webcam**: Chụp ảnh trực tiếp từ webcam để nhận dạng
        
        #### 2. Chế độ video tải lên
        - **Upload video**: Tải lên video có định dạng MP4, AVI, MOV, MKV, v.v.
        - **Điều chỉnh tốc độ xử lý**: Chọn tốc độ xử lý (Chế độ nhanh / Chế độ chất lượng cao)
        - **Tùy chọn hiển thị**: Hiển thị kết quả theo từng frame hoặc video hoàn chỉnh
        - **Thanh điều khiển video**: Tạm dừng, tua đi, tua lại, v.v.
        
        #### 3. Chế độ video trực tiếp
        - **Bắt đầu**: Mở camera và bắt đầu phát hiện khuôn mặt
        - **Dừng**: Dừng quá trình nhận dạng và đóng camera
        - **Điều chỉnh độ phân giải**: Chọn độ phân giải camera phù hợp
        - **Tốc độ xử lý**: Điều chỉnh tốc độ xử lý khung hình (giá trị thấp hơn = xử lý nhiều frame hơn)
        
        #### Mẹo sử dụng:
        - **Ánh sáng**: Đảm bảo khuôn mặt được chiếu sáng tốt
        - **Góc nhìn**: Nên chọn góc thẳng hoặc nghiêng nhẹ để có kết quả tốt nhất
        - **Khoảng cách**: Khuôn mặt nên chiếm khoảng 10-15% khung hình để có kết quả tối ưu
        - **Chất lượng video**: Ưu tiên video có độ phân giải cao (720p trở lên) và ít nhiễu
        - **Đăng ký mặt mới**: Nếu khuôn mặt chưa được nhận dạng, sử dụng chức năng "Đăng ký khuôn mặt mới"
        
        #### Xử lý lỗi:
        - **Không nhận dạng được**: Đảm bảo khuôn mặt đã được đăng ký trong hệ thống
        - **Nhận dạng sai**: Cập nhật database với nhiều mẫu khuôn mặt hơn
        - **Lỗi xử lý video**: Thử chuyển đổi video sang MP4 hoặc giảm độ phân giải
        - **Không hiển thị camera**: Kiểm tra quyền truy cập camera trong trình duyệt
        """)
    
    # Load models
    face_detector = load_face_detector()
    face_recognizer = load_face_recognizer()
    
    # Choose input mode
    mode = st.radio("Chọn chế độ nhận dạng:", ["📸 Ảnh tĩnh", "🎬 Video tải lên", "🎥 Video trực tiếp"])
    
    # Process based on selected mode
    if mode == "📸 Ảnh tĩnh":
        # Image input methods
        input_method = st.radio("Chọn nguồn ảnh:", ["Upload ảnh", "Chụp từ webcam"])
        
        if input_method == "Chụp từ webcam":
            # Camera capture interface  
            picture = st.camera_input("Chụp ảnh khuôn mặt")
            
            if picture:
                # Convert to OpenCV format
                bytes_data = picture.getvalue()
                img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
                
                # Create two columns for original and result images
                col1, col2 = st.columns(2)
                
                # Display original image
                with col1:
                    st.subheader("Ảnh gốc")
                    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)
                
                # Process image
                with st.spinner("Đang nhận dạng khuôn mặt..."):
                    faces, aligned_faces = face_detector.detect(img)
                    
                    if len(faces) > 0:
                        names = []
                        scores = []
                        
                        # Recognize faces (limit to 5)
                        for face_img in aligned_faces[:5]:
                            name, score = face_recognizer.identify(face_img)
                            names.append(name)
                            scores.append(score)
                        
                        # Draw results
                        from utils.face_utils import draw_results
                        result_img = draw_results(img, faces[:5], names, scores)
                    else:
                        result_img = img  # No faces detected
                        names = []
                        scores = []
                    
                    # Display result
                    with col2:
                        st.subheader("Kết quả")
                        st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), use_container_width=True)
                    
                    # Display information below images
                    if names:
                        st.markdown("### Thông tin nhận dạng:")
                        for i, (name, score) in enumerate(zip(names, scores)):
                            status = "✅ Đã nhận dạng" if name != "Unknown" else "❌ Không nhận dạng được"
                            st.write(f"**Người {i+1}:** {name} ({score:.2f}) - {status}")
                    else:
                        st.warning("Không phát hiện khuôn mặt nào trong ảnh!")
        
        else:  # Upload image mode
            uploaded_file = st.file_uploader("Chọn ảnh chứa khuôn mặt", type=["jpg", "jpeg", "png"])
            
            if uploaded_file is not None:
                # Read image
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                
                # Create two columns for original and result images
                col1, col2 = st.columns(2)
                
                # Display original image
                with col1:
                    st.subheader("Ảnh gốc")
                    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)
                
                # Process image
                with st.spinner("Đang nhận dạng khuôn mặt..."):
                    faces, aligned_faces = face_detector.detect(img)
                    
                    if len(faces) > 0:
                        names = []
                        scores = []
                        
                        # Recognize faces (limit to 5)
                        for face_img in aligned_faces[:5]:
                            name, score = face_recognizer.identify(face_img)
                            names.append(name)
                            scores.append(score)
                        
                        # Draw results
                        from utils.face_utils import draw_results
                        result_img = draw_results(img, faces[:5], names, scores)
                    else:
                        result_img = img  # No faces detected
                        names = []
                        scores = []
                    
                    # Display result
                    with col2:
                        st.subheader("Kết quả")
                        st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), use_container_width=True)
                    
                    # Display information below images
                    if names:
                        st.markdown("### Thông tin nhận dạng:")
                        for i, (name, score) in enumerate(zip(names, scores)):
                            status = "✅ Đã nhận dạng" if name != "Unknown" else "❌ Không nhận dạng được"
                            st.write(f"**Người {i+1}:** {name} ({score:.2f}) - {status}")
                    else:
                        st.warning("Không phát hiện khuôn mặt nào trong ảnh!")
    
    elif mode == "🎬 Video tải lên":
        # Video upload section
        st.subheader("Nhận dạng khuôn mặt từ video")
        
        # Upload video file
        video_file = st.file_uploader("Tải lên video", type=["mp4", "mov", "avi", "mkv", "wmv"])
        
        if video_file is not None:
            # Save uploaded video to a temporary file
            temp_video_path = f"temp_video_{int(time.time())}.mp4"
            with open(temp_video_path, "wb") as f:
                f.write(video_file.getbuffer())
                
            # Get video info
            cap = cv2.VideoCapture(temp_video_path)
            if not cap.isOpened():
                st.error("Không thể đọc video. Vui lòng thử lại với định dạng khác.")
            else:
                # Get video properties
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                # Display video info
                st.info(f"Thông tin video: {frame_width}x{frame_height}, {fps:.1f} FPS, {frame_count} frames")
                
                # Video processing options
                col1, col2 = st.columns(2)
                with col1:
                    process_mode = st.radio(
                        "Chế độ xử lý:",
                        ["Nhanh (Skip frames)", "Chất lượng cao (Full frames)"],
                        index=0
                    )
                with col2:
                    result_mode = st.radio(
                        "Hiển thị kết quả:",
                        ["Từng frame", "Video hoàn chỉnh"],
                        index=1
                    )
                
                # Skip frames for faster processing
                if process_mode == "Nhanh (Skip frames)":
                    # For long videos, skip more frames
                    if frame_count > 1000:
                        frame_skip = 10
                    elif frame_count > 500:
                        frame_skip = 5
                    else:
                        frame_skip = 2
                else:
                    frame_skip = 1
                
                # Initialize face tracker for more stable results
                face_tracker = StableFaceTracker(max_distance=40.0, min_frames=2, max_missing_frames=8)
                
                # Process video button
                if st.button("Bắt đầu xử lý video", type="primary", use_container_width=True):
                    # Create progress bar and status containers
                    progress_container = st.container()
                    with progress_container:
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        time_remaining = st.empty()
                    
                    # Container for results
                    result_container = st.empty()
                    info_container = st.empty()
                    
                    # If full video output is selected, prepare output video
                    if result_mode == "Video hoàn chỉnh":
                        # Create temporary output video file
                        output_video_path = f"temp_output_{int(time.time())}.mp4"
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
                    
                    # Process video frames
                    frame_idx = 0
                    processed_frames = 0
                    all_results = []  # List to store [frame_idx, faces, names, scores]
                    
                    # Tracking variables for stats
                    total_faces_detected = 0
                    identified_faces = 0
                    unknown_faces = 0
                    people_detected = set()
                    people_frame_counts = {}  # Track how many frames each person appears in
                    
                    # Time tracking for ETA calculation
                    start_time = time.time()
                    frames_processed_for_time = 0
                    processing_fps = 0
                    
                    try:
                        while True:
                            ret, frame = cap.read()
                            if not ret:
                                break
                                
                            # Process every Nth frame
                            if frame_idx % frame_skip == 0:
                                # Calculate ETA based on current processing speed
                                frames_processed_for_time += 1
                                elapsed_time = time.time() - start_time
                                if elapsed_time > 0:
                                    processing_fps = frames_processed_for_time / elapsed_time
                                    remaining_frames = frame_count - frame_idx
                                    eta_seconds = remaining_frames / processing_fps if processing_fps > 0 else 0
                                    eta_text = f"ETA: {eta_seconds:.0f}s" if eta_seconds < 60 else f"ETA: {eta_seconds/60:.1f}m"
                                else:
                                    eta_text = "Calculating ETA..."
                                
                                # Update progress
                                progress = (frame_idx + 1) / frame_count
                                progress_bar.progress(progress)
                                status_text.text(f"Đang xử lý: {frame_idx + 1}/{frame_count} frames ({progress*100:.1f}%)")
                                time_remaining.text(f"{eta_text} — {processing_fps:.1f} FPS")
                                
                                # Face detection
                                faces, aligned_faces = face_detector.detect(frame)
                                
                                # Update face tracking
                                stable_faces = face_tracker.update(faces, frame_idx)
                                
                                # Identify stable faces
                                stable_names = []
                                stable_scores = []
                                
                                # Only process aligned faces if any were detected
                                if len(faces) > 0:
                                    total_faces_detected += len(faces)
                                    
                                    for face in stable_faces:
                                        if face['is_stable']:
                                            # Find corresponding aligned face (optimized matching)
                                            best_face_img = None
                                            min_dist = float('inf')
                                            
                                            for i, detected_face in enumerate(faces):
                                                # Simplified distance calculation
                                                if i < len(aligned_faces):
                                                    dist = np.sum(np.abs(detected_face[:4] - face['box']))
                                                    if dist < min_dist:
                                                        min_dist = dist
                                                        best_face_img = aligned_faces[i]
                                            
                                            # Identify face if found
                                            if best_face_img is not None:
                                                name, score = face_recognizer.identify(best_face_img)
                                                stable_names.append(name)
                                                stable_scores.append(score)
                                                
                                                # Update stats
                                                if name != "Unknown":
                                                    identified_faces += 1
                                                    people_detected.add(name)
                                                    # Count appearances
                                                    if name in people_frame_counts:
                                                        people_frame_counts[name] += 1
                                                    else:
                                                        people_frame_counts[name] = 1
                                                else:
                                                    unknown_faces += 1
                                
                                # Draw results
                                result_frame = draw_stable_results(frame, stable_faces, stable_names, stable_scores)
                                
                                # Add frame number for reference
                                cv2.putText(
                                    result_frame, 
                                    f"Frame: {frame_idx}/{frame_count}", 
                                    (10, frame_height - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
                                )
                                
                                # Store results
                                all_results.append([frame_idx, stable_faces, stable_names, stable_scores, result_frame])
                                
                                # Display result if in frame mode
                                if result_mode == "Từng frame" and processed_frames % 5 == 0:  # Only update display every 5 processed frames
                                    result_container.image(cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                                    
                                    # Display face info
                                    if stable_names:
                                        info_text = "**Khuôn mặt đã phát hiện:**\n\n"
                                        for name, score in zip(stable_names, stable_scores):
                                            status = "✅ Đã nhận dạng" if name != "Unknown" else "❌ Chưa nhận dạng"
                                            info_text += f"- {name} ({score:.2f}) - {status}\n"
                                        info_container.markdown(info_text)
                                
                                # Save to output video if in video mode
                                if result_mode == "Video hoàn chỉnh":
                                    out.write(result_frame)
                                
                                processed_frames += 1
                                
                            frame_idx += 1
                    
                    except Exception as e:
                        st.error(f"Lỗi khi xử lý video: {str(e)}")
                    finally:
                        # Close video capture and writer
                        cap.release()
                        if result_mode == "Video hoàn chỉnh":
                            out.release()
                        
                        # Clear progress indicators
                        progress_container.empty()
                        
                        # Calculate processing stats
                        total_time = time.time() - start_time
                        processing_fps = processed_frames / total_time if total_time > 0 else 0
                        
                        # Display stats
                        st.success(f"✅ Đã xử lý xong {processed_frames} frames trong {total_time:.1f} giây ({processing_fps:.1f} FPS)")
                        
                        stats_col1, stats_col2 = st.columns(2)
                        with stats_col1:
                            st.markdown("### Thống kê nhận dạng")
                            st.markdown(f"- Tổng số khuôn mặt đã phát hiện: **{total_faces_detected}**")
                            st.markdown(f"- Khuôn mặt đã nhận dạng: **{identified_faces}** ({identified_faces/max(total_faces_detected, 1)*100:.1f}%)")
                            st.markdown(f"- Khuôn mặt chưa nhận dạng: **{unknown_faces}** ({unknown_faces/max(total_faces_detected, 1)*100:.1f}%)")
                            
                        with stats_col2:
                            st.markdown("### Người xuất hiện trong video")
                            if people_detected:
                                # Sắp xếp theo số lần xuất hiện nhiều nhất
                                sorted_people = sorted(people_frame_counts.items(), key=lambda x: x[1], reverse=True)
                                for person, count in sorted_people:
                                    frames_percentage = count / processed_frames * 100
                                    st.markdown(f"- **{person}**: {count} frames ({frames_percentage:.1f}%)")
                            else:
                                st.markdown("Không có người nào được nhận dạng")
                        
                        # Show final result
                        if result_mode == "Video hoàn chỉnh":
                            # Display processed video
                            st.markdown("### Video kết quả")
                            
                            try:
                                # Đảm bảo video được đóng trước khi đọc lại
                                if 'out' in locals():
                                    out.release()
                                    
                                # Kiểm tra xem file có tồn tại và có kích thước không
                                if os.path.exists(output_video_path) and os.path.getsize(output_video_path) > 0:
                                    # Thêm codec cho đúng định dạng web
                                    temp_web_path = f"temp_web_{int(time.time())}.mp4"
                                    os.system(f"ffmpeg -y -i {output_video_path} -vcodec libx264 -pix_fmt yuv420p {temp_web_path}")
                                    
                                    if os.path.exists(temp_web_path) and os.path.getsize(temp_web_path) > 0:
                                        # Đọc video đã xử lý với đúng định dạng cho web
                                        video_bytes = open(temp_web_path, 'rb').read()
                                        
                                        # Hiển thị video sử dụng component có sẵn của Streamlit
                                        st.video(video_bytes)
                                        
                                        # Thêm nút tải về với style rõ ràng
                                        st.download_button(
                                            label="📥 Tải video kết quả",
                                            data=video_bytes,
                                            file_name=f"face_recognition_{int(time.time())}.mp4",
                                            mime="video/mp4",
                                            use_container_width=True
                                        )
                                        
                                        # Xóa file tạm
                                        try:
                                            os.remove(temp_web_path)
                                        except:
                                            pass
                                    else:
                                        st.error("Không thể chuyển đổi video sang định dạng web. Hãy thử lại.")
                                        st.info("Bạn có thể cần cài đặt ffmpeg: `pip install ffmpeg-python`")
                                else:
                                    st.error("Video kết quả không tồn tại hoặc rỗng. Vui lòng thử lại quá trình xử lý.")
                            except Exception as e:
                                st.error(f"Lỗi khi hiển thị video: {str(e)}")
                                st.info("Vui lòng thử lại với video khác hoặc chọn chế độ hiển thị từng frame.")
                                
                            # Clean up temporary files
                            try:
                                os.remove(output_video_path)
                            except:
                                pass
                        else:
                            # Display the highlights
                            st.markdown("### Các frame đáng chú ý")
                            
                            # Find frames with the most recognized faces
                            highlight_frames = []
                            for result in all_results:
                                frame_idx, faces, names, scores, frame = result
                                recognized_count = sum(1 for name in names if name != "Unknown")
                                if recognized_count > 0:
                                    highlight_frames.append((recognized_count, frame_idx, frame))
                            
                            # Sort by recognition count (most first)
                            highlight_frames.sort(reverse=True)
                            
                            # Display top highlights (max 5)
                            max_highlights = min(5, len(highlight_frames))
                            if max_highlights > 0:
                                highlight_cols = st.columns(max_highlights)
                                for i in range(max_highlights):
                                    if i < len(highlight_frames):
                                        count, frame_idx, frame = highlight_frames[i]
                                        with highlight_cols[i]:
                                            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                                            st.caption(f"Frame {frame_idx} - {count} người")
                            else:
                                st.warning("Không tìm thấy frame nào với khuôn mặt được nhận dạng")
                            
                    # Clean up temporary file
                    try:
                        os.remove(temp_video_path)
                    except:
                        pass
    elif mode == "🎥 Video trực tiếp":
        st.markdown("### 🎥 Video trực tiếp từ camera")
        
        # Performance settings
        col1, col2 = st.columns(2)
        with col1:
            resolution = st.selectbox("Độ phân giải:", 
                                    ["640x480", "800x600", "1280x720"],
                                    index=0)
        with col2:
            process_rate = st.slider("Tốc độ xử lý:", 1, 6, 2, 1)
        
        # Parse resolution
        w, h = map(int, resolution.split('x'))
        
        # Control buttons
        col1, col2 = st.columns([1, 4])
        with col1:
            start_button = st.button("Bắt đầu", type="primary", use_container_width=True)
        with col2:
            stop_button = st.button("Dừng", use_container_width=True, key="stop_button")
        
        # Initialize session state
        if 'video_running' not in st.session_state:
            st.session_state.video_running = False
        
        # Initialize face tracker (faster settings)
        face_tracker = StableFaceTracker(max_distance=40.0, min_frames=2, max_missing_frames=8)
        
        # Video display area
        video_placeholder = st.empty()
        info_placeholder = st.empty()
        
        # Start video stream
        if start_button:
            st.session_state.video_running = True
        
        if stop_button:
            st.session_state.video_running = False
        
        # Video processing loop
        if st.session_state.video_running:
            # Hiển thị loading state
            video_placeholder.markdown("### ⏳ Đang mở camera...")
            
            # Try different camera indices
            cap = None
            for idx in range(3):  # Try index 0, 1, 2
                cap = cv2.VideoCapture(idx)
                if cap.isOpened():
                    break
                cap.release()
            
            if cap is None or not cap.isOpened():
                st.error("Không thể mở camera. Vui lòng kiểm tra lại kết nối camera.")
            else:
                # Set camera resolution
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                # Hiển thị thành công
                video_placeholder.markdown("✅ Camera đã sẵn sàng")
                time.sleep(0.5)  # Hiển thị thông báo trong 0.5 giây
                
                # FPS counter and frame processing counter
                fps_start_time = time.time()
                fps_counter = 0
                current_fps = 0
                frame_num = 0
                
                # Initialize variable to track last processed frame
                result_frame = None
                names = []
                scores = []
                
                while st.session_state.video_running:
                    ret, frame = cap.read()
                    
                    if not ret:
                        st.error("Không thể đọc khung hình từ camera!")
                        break
                    
                    # Flip frame horizontally for mirror effect
                    frame = cv2.flip(frame, 1)
                    
                    # Process every Nth frame based on speed setting
                    if frame_num % process_rate == 0:
                        # Detect faces
                        faces, aligned_faces = face_detector.detect(frame)
                        
                        # Update face tracker
                        stable_faces = face_tracker.update(faces, frame_num)
                        
                        # Only recognize stable faces
                        stable_names = []
                        stable_scores = []
                        
                        for face in stable_faces:
                            if face['is_stable']:
                                # Find corresponding aligned face (optimized matching)
                                best_face_img = None
                                min_dist = float('inf')
                                
                                for i, detected_face in enumerate(faces):
                                    # Simplified distance calculation
                                    if i < len(aligned_faces):
                                        dist = np.sum(np.abs(detected_face[:4] - face['box']))
                                        if dist < min_dist:
                                            min_dist = dist
                                            best_face_img = aligned_faces[i]
                                
                                if best_face_img is not None:
                                    name, score = face_recognizer.identify(best_face_img)
                                    stable_names.append(name)
                                    stable_scores.append(score)
                        
                        # Draw stable results (optimized function)
                        result_frame = draw_stable_results(frame, stable_faces, stable_names, stable_scores)
                        names = stable_names
                        scores = stable_scores
                    
                    frame_num += 1
                    
                    # Use the last processed frame if available, otherwise use raw frame
                    display_frame = result_frame if result_frame is not None else frame
                    
                    # Calculate FPS
                    fps_counter += 1
                    if time.time() - fps_start_time >= 1.0:
                        current_fps = fps_counter
                        fps_counter = 0
                        fps_start_time = time.time()
                    
                    # Add FPS text to frame (simplified drawing)
                    cv2.putText(display_frame, f"FPS: {current_fps}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # Convert to RGB for display only once
                    display_frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                    
                    # Display frame
                    video_placeholder.image(display_frame_rgb, channels="RGB", use_container_width=True)
                    
                    # Display recognition info (only show if needed)
                    if frame_num % (process_rate * 2) == 0:  # Update info less frequently
                        if names:
                            with info_placeholder:
                                # Count unique faces (only show green boxes count)
                                stable_count = sum(1 for face in stable_faces if face['is_stable'])
                                actual_people = min(stable_count, len(names))
                                
                                cols = st.columns(min(actual_people, 3))
                                for i in range(actual_people):
                                    if i < len(names) and i < len(scores):
                                        with cols[i % 3]:
                                            status = "✅ Đã nhận dạng" if names[i] != "Unknown" else "❌ Không nhận dạng được"
                                            st.markdown(f"""
                                            **Người {i+1}**  
                                            {names[i]}  
                                            Score: {scores[i]:.2f}  
                                            {status}
                                            """)
                        else:
                            info_placeholder.markdown("*Đang phát hiện khuôn mặt...*")
                    
                    # Check if stop button was pressed
                    if not st.session_state.video_running:
                        break
                
                # Release camera
                cap.release()
                video_placeholder.markdown("**Camera đã được đóng**")
        
        elif not st.session_state.video_running and not start_button:
            video_placeholder.markdown("""
            ## 🎥 Chế độ Video Trực Tiếp
            
            **Tính năng:**
            - Hiển thị khung khuôn mặt ổn định, không chớp nháy
            - Tự động nhận dạng khuôn mặt trong thời gian thực
            - Hiển thị FPS để theo dõi hiệu suất
            
            **Tùy chọn hiệu suất:**
            - Độ phân giải thấp = nhanh hơn
            - Tốc độ xử lý cao = xử lý nhiều frame hơn/giây
            
            Nhấn **Bắt đầu** để mở camera.
            """)