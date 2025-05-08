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
        # Video upload section with improved explanation
        st.subheader("Nhận dạng khuôn mặt từ video")
        
        # Add helpful explanation
        st.markdown("""
        Tính năng này cho phép nhận dạng khuôn mặt từ file video tải lên.
        Quá trình xử lý sẽ diễn ra theo thời gian thực - bạn sẽ thấy kết quả ngay khi video đang chạy!
        """)
        
        # Upload video file with clearer instructions
        video_file = st.file_uploader(
            "Tải lên video chứa khuôn mặt cần nhận dạng", 
            type=["mp4", "mov", "avi", "mkv", "wmv"],
            help="Hỗ trợ các định dạng: MP4, MOV, AVI, MKV, WMV. Nên sử dụng video MP4 để có hiệu suất tốt nhất."
        )
        
        if video_file is not None:
            # Process video in realtime mode
            process_video_realtime(video_file, face_detector, face_recognizer)
            
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
            
def process_video_realtime(video_file, face_detector, face_recognizer):
    """
    Xử lý video theo thời gian thực - không cần đợi xử lý tất cả các frame
    """
    import tempfile
    import os
    import threading
    import queue
    import time
    from datetime import timedelta
    
    # Tạo file tạm
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmpfile:
        temp_video_path = tmpfile.name
        # Lưu video vào file tạm
        tmpfile.write(video_file.getbuffer())
    
    # Mở video
    cap = cv2.VideoCapture(temp_video_path)
    if not cap.isOpened():
        st.error("Không thể đọc video. Vui lòng thử với định dạng khác.")
        try:
            os.remove(temp_video_path)
        except:
            pass
        return
    
    # Lấy thông tin video
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = frame_count / fps if fps > 0 else 0
    
    # Hiển thị thông tin video
    st.info(f"Thông tin video: {frame_width}x{frame_height}, {fps:.1f} FPS, {frame_count} frames, thời lượng: {timedelta(seconds=duration)}")
    
    # Điều chỉnh độ nhạy và tốc độ xử lý
    col1, col2 = st.columns(2)
    with col1:
        confidence_threshold = st.slider(
            "Ngưỡng nhận diện:", 
            min_value=0.5, 
            max_value=0.95, 
            value=0.7, 
            step=0.05,
            help="Điều chỉnh độ nhạy khi nhận diện (cao hơn = ít nhận diện sai hơn)"
        )
    with col2:
        skip_frames = st.slider(
            "Tốc độ xử lý:", 
            min_value=1, 
            max_value=10, 
            value=2, 
            step=1,
            help="Số frame bỏ qua khi xử lý (cao hơn = nhanh hơn nhưng mất một số chi tiết)"
        )
    
    # Khởi tạo face tracker
    face_tracker = StableFaceTracker(
        max_distance=40.0,
        min_frames=2,
        max_missing_frames=10
    )
    
    # Thông tin theo dõi
    stats = {
        'total_faces': 0,
        'identified_faces': 0,
        'unknown_faces': 0,
        'people_detected': set(),
        'people_frames': {}
    }
    
    # Queue để trao đổi dữ liệu giữa các luồng
    frame_queue = queue.Queue(maxsize=30)  # Buffer 30 frames
    result_queue = queue.Queue(maxsize=30)
    stop_event = threading.Event()
    
    # Placeholder hiển thị
    video_placeholder = st.empty()
    progress_bar = st.progress(0.0)
    info_placeholder = st.empty()
    stats_placeholder = st.empty()
    
    # Thread xử lý frame
    def process_frames():
        frame_idx = 0
        processed_idx = 0
        
        while not stop_event.is_set():
            try:
                if frame_queue.empty():
                    time.sleep(0.01)
                    continue
                
                frame, current_pos = frame_queue.get()
                frame_idx += 1
                
                # Chỉ xử lý mỗi N frame theo skip_frames
                if frame_idx % skip_frames == 0:
                    # Face detection
                    start_time = time.time()
                    faces, aligned_faces = face_detector.detect(frame)
                    
                    # Update face tracking
                    stable_faces = face_tracker.update(faces, frame_idx)
                    
                    # Nhận diện các khuôn mặt ổn định
                    stable_names = []
                    stable_scores = []
                    
                    if len(faces) > 0:
                        stats['total_faces'] += len(faces)
                        
                        for face in stable_faces:
                            if face['is_stable']:
                                # Tìm aligned face tương ứng
                                best_face_img = None
                                min_dist = float('inf')
                                
                                for i, detected_face in enumerate(faces):
                                    if i < len(aligned_faces):
                                        dist = np.sum(np.abs(detected_face[:4] - face['box']))
                                        if dist < min_dist:
                                            min_dist = dist
                                            best_face_img = aligned_faces[i]
                                
                                if best_face_img is not None:
                                    name, score = face_recognizer.identify(best_face_img)
                                    
                                    # Áp dụng ngưỡng tin cậy
                                    if score < confidence_threshold:
                                        name = "Unknown"
                                        stats['unknown_faces'] += 1
                                    else:
                                        stats['identified_faces'] += 1
                                        stats['people_detected'].add(name)
                                        # Đếm số frame mỗi người xuất hiện
                                        stats['people_frames'][name] = stats['people_frames'].get(name, 0) + 1
                                    
                                    stable_names.append(name)
                                    stable_scores.append(score)
                    
                    # Vẽ kết quả
                    result_frame = draw_stable_results(frame, stable_faces, stable_names, stable_scores)
                    
                    # Tính thời gian xử lý
                    process_time = time.time() - start_time
                    fps_text = f"Processing: {1/process_time:.1f} FPS"
                    
                    # Thêm thông tin vào frame
                    cv2.putText(
                        result_frame, 
                        fps_text, 
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                    )
                    
                    # Thêm vị trí frame
                    position_text = f"Frame: {frame_idx}/{frame_count} ({current_pos*100:.0f}%)"
                    cv2.putText(
                        result_frame, 
                        position_text, 
                        (10, frame_height - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                    )
                    
                    # Gửi kết quả vào queue
                    result_queue.put((result_frame, stable_names, stable_scores, current_pos))
                    processed_idx += 1
                
                frame_queue.task_done()
                
            except Exception as e:
                print(f"Lỗi xử lý frame: {e}")
                if frame_queue.qsize() > 0:
                    frame_queue.task_done()
                time.sleep(0.1)
    
    # Thread đọc frame
    def read_frames():
        frame_idx = 0
        try:
            while cap.isOpened() and not stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Tính vị trí tương đối
                current_pos = cap.get(cv2.CAP_PROP_POS_FRAMES) / frame_count
                
                # Đưa frame vào queue để xử lý, với kiểm tra để tránh tắc nghẽn
                if not frame_queue.full():
                    frame_queue.put((frame, current_pos))
                else:
                    # Nếu queue đầy, đợi một chút
                    time.sleep(0.01)
                
                frame_idx += 1
                
        except Exception as e:
            print(f"Lỗi đọc frame: {e}")
        finally:
            # Đánh dấu đã đọc xong
            stop_event.set()
            cap.release()
    
    # Nút điều khiển
    st.markdown("### Điều khiển xử lý video")
    col1, col2 = st.columns(2)
    
    with col1:
        start_button = st.button("▶️ Bắt đầu xử lý", type="primary", use_container_width=True)
    with col2:
        stop_button = st.button("⏹️ Dừng xử lý", use_container_width=True)
    
    if start_button:
        # Khởi động các thread
        process_thread = threading.Thread(target=process_frames)
        read_thread = threading.Thread(target=read_frames)
        
        process_thread.daemon = True
        read_thread.daemon = True
        
        # Bắt đầu xử lý
        process_thread.start()
        read_thread.start()
        
        start_time = time.time()
        frame_count_displayed = 0
        
        # Hiển thị kết quả theo thời gian thực
        st.markdown("### Video đang xử lý")
        
        # Vòng lặp hiển thị kết quả
        try:
            while not stop_event.is_set() or not result_queue.empty():
                if stop_button:
                    stop_event.set()
                    st.warning("Đang dừng xử lý...")
                    break
                
                if not result_queue.empty():
                    result_frame, names, scores, pos = result_queue.get()
                    frame_count_displayed += 1
                    
                    # Cập nhật video và thanh tiến trình
                    progress_bar.progress(pos)
                    video_placeholder.image(
                        cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB),
                        use_container_width=True
                    )
                    
                    # Tính FPS hiển thị
                    elapsed = time.time() - start_time
                    display_fps = frame_count_displayed / elapsed if elapsed > 0 else 0
                    
                    # Hiển thị thông tin nhận diện hiện tại
                    if names:
                        info_text = f"**Tốc độ hiển thị: {display_fps:.1f} FPS | Khuôn mặt nhận diện được:**\n\n"
                        for name, score in zip(names, scores):
                            status = "✅" if name != "Unknown" else "❌"
                            info_text += f"{status} {name} ({score:.2f}) "
                        info_placeholder.markdown(info_text)
                    
                    # Hiển thị thống kê tổng hợp
                    if frame_count_displayed % 10 == 0:  # Cập nhật mỗi 10 frame
                        stats_text = f"""
                        ### Thống kê xử lý:
                        - Đã xử lý: {frame_count_displayed} frames ({display_fps:.1f} FPS)
                        - Khuôn mặt phát hiện: {stats['total_faces']}
                        - Khuôn mặt nhận diện: {stats['identified_faces']}
                        - Không nhận diện được: {stats['unknown_faces']}
                        - Số người nhận diện được: {len(stats['people_detected'])}
                        """
                        
                        # Hiển thị top 3 người xuất hiện nhiều nhất
                        if stats['people_frames']:
                            stats_text += "\n\n**Top người xuất hiện nhiều nhất:**\n"
                            sorted_people = sorted(stats['people_frames'].items(), key=lambda x: x[1], reverse=True)
                            for person, count in sorted_people[:3]:
                                stats_text += f"- {person}: {count} frames\n"
                                
                        stats_placeholder.markdown(stats_text)
                    
                    result_queue.task_done()
                else:
                    # Nếu không có kết quả mới, đợi một chút
                    time.sleep(0.01)
        
        except Exception as e:
            st.error(f"Lỗi hiển thị kết quả: {e}")
        finally:
            # Đảm bảo dừng tất cả luồng
            stop_event.set()
            
            # Đợi các thread kết thúc
            if 'process_thread' in locals() and process_thread.is_alive():
                process_thread.join(timeout=1.0)
            if 'read_thread' in locals() and read_thread.is_alive():
                read_thread.join(timeout=1.0)
            
            # Tính thời gian chạy
            run_time = time.time() - start_time
            
            # Thông báo kết thúc
            st.success(f"Đã hoàn thành xử lý {frame_count_displayed} frames trong {run_time:.1f} giây!")
            
            # Hiển thị thống kê cuối cùng
            st.subheader("Kết quả nhận diện")
            
            # Hiển thị các người được nhận diện 
            if stats['people_detected']:
                st.markdown("### Người xuất hiện trong video:")
                
                # Sắp xếp theo số lần xuất hiện
                if stats['people_frames']:
                    people_cols = st.columns(min(3, len(stats['people_frames'])))
                    sorted_people = sorted(stats['people_frames'].items(), key=lambda x: x[1], reverse=True)
                    
                    for i, (person, frames) in enumerate(sorted_people):
                        with people_cols[i % 3]:
                            frame_percent = frames / frame_count_displayed * 100
                            st.metric(
                                label=person,
                                value=f"{frames} frames",
                                delta=f"{frame_percent:.1f}%"
                            )
            else:
                st.warning("Không nhận diện được người nào trong video.")
    
    # Dọn dẹp
    try:
        os.remove(temp_video_path)
    except:
        pass