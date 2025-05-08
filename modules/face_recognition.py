import streamlit as st
import cv2
import numpy as np
import time
import tempfile
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

@st.cache_resource
def load_face_tracker():
    from utils.face_utils import StableFaceTracker
    return StableFaceTracker(max_distance=40.0, min_frames=2, max_missing_frames=8)

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
        
        **Ứng dụng thực tế:**
        - Hệ thống bảo mật và kiểm soát truy cập
        - Hệ thống điểm danh tự động
        - Trải nghiệm cá nhân hóa trong các hệ thống thông minh
        - Xác thực danh tính không tiếp xúc
        """)
            
    with st.expander("📋 Hướng dẫn sử dụng", expanded=False):
        st.markdown("""
        ### Hướng dẫn sử dụng
        
        #### 1. Chế độ ảnh tĩnh
        - **Upload ảnh**: Tải lên ảnh chứa khuôn mặt cần nhận dạng
        - **Chụp từ webcam**: Chụp ảnh trực tiếp từ webcam để nhận dạng
        
        #### 2. Chế độ video
        - **Video trực tiếp**: Sử dụng webcam để nhận dạng khuôn mặt theo thời gian thực
        - **Upload video**: Tải lên và phân tích video có sẵn để nhận dạng khuôn mặt
        
        #### 3. Tùy chỉnh xử lý video
        - **Tốc độ xử lý**: Điều chỉnh tốc độ xử lý khung hình để cân bằng giữa hiệu suất và độ chính xác
        - **Độ phân giải**: Chọn độ phân giải phù hợp với nhu cầu
        - **Phân tích video**: Tùy chỉnh mức độ chi tiết trong phân tích video
        
        #### Mẹo sử dụng:
        - **Ánh sáng**: Đảm bảo khuôn mặt được chiếu sáng tốt
        - **Góc nhìn**: Hướng mặt thẳng vào camera để có kết quả tốt nhất
        - **Khoảng cách**: Đứng cách camera khoảng 0.5-1m
        - **Đăng ký mặt mới**: Nếu khuôn mặt chưa được nhận dạng, sử dụng chức năng "Đăng ký khuôn mặt mới"
        
        #### Xử lý lỗi:
        - **Không nhận dạng được**: Đảm bảo khuôn mặt đã được đăng ký trong hệ thống
        - **Nhận dạng sai**: Cập nhật database với nhiều mẫu khuôn mặt hơn
        - **Không hiển thị camera**: Kiểm tra quyền truy cập camera trong trình duyệt
        """)
    
    # Choose input mode
    mode = st.radio("Chọn chế độ nhận dạng:", ["📸 Ảnh tĩnh", "🎥 Video trực tiếp", "📹 Video upload"])
    
    # Load models
    face_detector = load_face_detector()
    face_recognizer = load_face_recognizer()
    
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
                            name, score = face_recognizer.identify(face_img, 0.6)  # Set threshold to 0.6
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
                    
                    # Display information below images - only show high confidence results
                    has_high_confidence = False
                    if names:
                        st.markdown("### Thông tin nhận dạng:")
                        for i, (name, score) in enumerate(zip(names, scores)):
                            if score >= 0.6:  # Only show results with confidence >= 0.6
                                has_high_confidence = True
                                status = "✅ Đã nhận dạng" if name != "Unknown" else "❌ Không nhận dạng được"
                                st.write(f"**Người {i+1}:** {name} ({score:.2f}) - {status}")
                        
                        if not has_high_confidence:
                            st.info("Không có khuôn mặt nào được nhận dạng với độ tin cậy > 0.6")
                    else:
                        st.warning("Không phát hiện khuôn mặt nào trong ảnh!")
        
        else:  # Upload image mode
            uploaded_file = st.file_uploader("Chọn ảnh chứa khuôn mặt", type=["jpg", "jpeg", "png", "bmp"])
            
            if uploaded_file is not None:
                # Read image
                try:
                    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    
                    # Check if image was loaded successfully
                    if img is None:
                        st.error("Không thể đọc ảnh. Vui lòng thử file khác.")
                    else:
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
                                    name, score = face_recognizer.identify(face_img, 0.6)  # Threshold 0.6
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
                            
                            # Display information below images - only show high confidence results
                            has_high_confidence = False
                            if names:
                                st.markdown("### Thông tin nhận dạng:")
                                for i, (name, score) in enumerate(zip(names, scores)):
                                    if score >= 0.6:  # Only show results with confidence >= 0.6
                                        has_high_confidence = True
                                        status = "✅ Đã nhận dạng" if name != "Unknown" else "❌ Không nhận dạng được"
                                        st.write(f"**Người {i+1}:** {name} ({score:.2f}) - {status}")
                                
                                if not has_high_confidence:
                                    st.info("Không có khuôn mặt nào được nhận dạng với độ tin cậy > 0.6")
                            else:
                                st.warning("Không phát hiện khuôn mặt nào trong ảnh!")
                except Exception as e:
                    st.error(f"Lỗi khi xử lý ảnh: {str(e)}")
    
    elif mode == "🎥 Video trực tiếp":
        st.markdown("### 🎥 Video trực tiếp từ camera")
        
        # Performance settings
        col1, col2 = st.columns(2)
        with col1:
            resolution = st.selectbox("Độ phân giải:", 
                                    ["640x480", "800x600", "1280x720"],
                                    index=0)
        with col2:
            process_rate = st.slider("Tốc độ xử lý:", 1, 6, 2, 1,
                                   help="Giá trị càng cao, tốc độ xử lý càng nhanh nhưng có thể bỏ lỡ một số khung hình")
        
        # Parse resolution
        w, h = map(int, resolution.split('x'))
        
        # Control buttons
        col1, col2 = st.columns([1, 4])
        with col1:
            start_button = st.button("Bắt đầu", type="primary", use_container_width=True)
        with col2:
            stop_button = st.button("Dừng", use_container_width=True)
        
        # Initialize session state
        if 'video_running' not in st.session_state:
            st.session_state.video_running = False
        
        # Initialize face tracker (faster settings)
        face_tracker = load_face_tracker()
        
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
                
                # Performance optimization - use thread settings
                import threading
                if "last_process_time" not in st.session_state:
                    st.session_state.last_process_time = 0
                    
                process_interval = 0.05  # 50ms minimum between processing
                last_info_update = 0  # Track when we last updated info display
                
                try:
                    while st.session_state.video_running:
                        ret, frame = cap.read()
                        
                        if not ret:
                            st.error("Không thể đọc khung hình từ camera!")
                            break
                        
                        # Flip frame horizontally for mirror effect
                        frame = cv2.flip(frame, 1)
                        
                        # Calculate current FPS
                        fps_counter += 1
                        current_time = time.time()
                        if current_time - fps_start_time >= 1.0:
                            current_fps = fps_counter
                            fps_counter = 0
                            fps_start_time = current_time
                        
                        # Process frame if enough time has passed or it's the first frame
                        current_time = time.time()
                        should_process = (
                            frame_num % process_rate == 0 and 
                            current_time - st.session_state.last_process_time >= process_interval
                        )
                        
                        if should_process:
                            st.session_state.last_process_time = current_time
                            
                            # Run face detection in a background thread
                            def process_frame(frame_to_process, frame_number):
                                # Detect faces
                                faces, aligned_faces = face_detector.detect(frame_to_process)
                                
                                # Update face tracker
                                stable_faces = face_tracker.update(faces, frame_number)
                                
                                # Recognize stable faces only - OPTIMIZED to avoid unnecessary processing
                                stable_indices = []
                                stable_names = []
                                stable_scores = []
                                aligned_face_map = {}
                                
                                # Create a lookup map for aligned faces based on approximate position
                                if len(aligned_faces) > 0 and len(faces) > 0:
                                    for i, face_box in enumerate(faces):
                                        if i < len(aligned_faces):
                                            # Use face box center as key
                                            center_x = int(face_box[0] + face_box[2] / 2)
                                            center_y = int(face_box[1] + face_box[3] / 2)
                                            aligned_face_map[(center_x, center_y)] = aligned_faces[i]
                                
                                # Only process stable faces
                                for i, face in enumerate(stable_faces):
                                    if face['is_stable']:
                                        stable_indices.append(i)
                                        box = face['box']
                                        
                                        # Find closest aligned face
                                        center_x = int(box[0] + box[2] / 2)
                                        center_y = int(box[1] + box[3] / 2)
                                        
                                        # Find closest aligned face in map within reasonable distance
                                        best_dist = float('inf')
                                        best_face = None
                                        for (x, y), aligned_face in aligned_face_map.items():
                                            dist = abs(x - center_x) + abs(y - center_y)
                                            if dist < best_dist and dist < 50:  # 50 pixel threshold
                                                best_dist = dist
                                                best_face = aligned_face
                                        
                                        # If found, recognize the face
                                        if best_face is not None:
                                            name, score = face_recognizer.identify(best_face, 0.6)
                                            stable_names.append(name)
                                            stable_scores.append(score)
                                        else:
                                            # If no aligned face found, use Unknown
                                            stable_names.append("Unknown")
                                            stable_scores.append(0.0)
                                
                                # Draw results
                                from utils.face_utils import draw_stable_results
                                processed_frame = draw_stable_results(
                                    frame_to_process, stable_faces, stable_names, stable_scores
                                )
                                
                                # Add FPS counter
                                cv2.putText(
                                    processed_frame, f"FPS: {current_fps}", 
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
                                )
                                
                                return processed_frame, stable_names, stable_scores
                            
                            # Process current frame
                            result_frame, names, scores = process_frame(frame.copy(), frame_num)
                        
                        frame_num += 1
                        
                        # Always display the last processed frame if available
                        display_frame = result_frame if result_frame is not None else frame
                        
                        # Add FPS text to raw frame if no processed frame
                        if result_frame is None:
                            cv2.putText(
                                display_frame, f"FPS: {current_fps}", 
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
                            )
                        
                        # Convert to RGB for display
                        display_frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                        
                        # Display frame
                        video_placeholder.image(display_frame_rgb, channels="RGB", use_container_width=True)
                        
                        # Update recognition info (only update occasionally to avoid UI slowdown)
                        current_time = time.time()
                        if current_time - last_info_update > 0.5:  # Update every 0.5 seconds
                            last_info_update = current_time
                            
                            # Filter by confidence threshold 0.6
                            high_conf_indices = [i for i, score in enumerate(scores) if score >= 0.6]
                            
                            # Display information based on high confidence detections
                            with info_placeholder:
                                if high_conf_indices:
                                    st.markdown("### Người được nhận dạng:")
                                    cols = st.columns(min(len(high_conf_indices), 3))
                                    for idx, i in enumerate(high_conf_indices):
                                        with cols[idx % 3]:
                                            status = "✅ Đã nhận dạng" if names[i] != "Unknown" else "❌ Không nhận dạng được"
                                            st.markdown(f"""
                                            **Người {idx+1}**  
                                            {names[i]}  
                                            Score: {scores[i]:.2f}  
                                            {status}
                                            """)
                                else:
                                    if names:
                                        st.info("Không có khuôn mặt nào được nhận dạng với độ tin cậy > 0.6")
                                    else:
                                        st.markdown("*Đang tìm kiếm khuôn mặt...*")
                        
                        # Prevent CPU overuse and reduce UI bottlenecks
                        time.sleep(0.01)
                        
                        # Check if stop button was pressed
                        if not st.session_state.video_running:
                            break
                        
                except Exception as e:
                    st.error(f"Lỗi trong quá trình xử lý video: {str(e)}")
                finally:
                    # Release camera
                    if cap is not None:
                        cap.release()
                    video_placeholder.markdown("**Camera đã được đóng**")
        
        elif not st.session_state.video_running and not start_button:
            video_placeholder.markdown("""
            ## 🎥 Chế độ Video Trực Tiếp
            
            **Tính năng:**
            - Hiển thị khung khuôn mặt ổn định, không chớp nháy
            - Tự động nhận dạng khuôn mặt trong thời gian thực
            - Hiển thị FPS để theo dõi hiệu suất
            - Chỉ hiển thị kết quả có độ tin cậy > 0.6
            
            **Tùy chọn hiệu suất:**
            - Độ phân giải thấp = nhanh hơn
            - Tốc độ xử lý cao = xử lý ít frame hơn, mượt hơn
            
            Nhấn **Bắt đầu** để mở camera.
            """)
    
    elif mode == "📹 Video upload":
        st.markdown("### 📹 Phân tích video từ file")
        
        # Upload video file
        uploaded_video = st.file_uploader("Chọn video để phân tích", type=['mp4', 'avi', 'mov', 'mkv'])
        
        if uploaded_video is not None:
            # Save uploaded video to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
                temp_file.write(uploaded_video.read())
                video_path = temp_file.name
            
            # Video settings
            st.subheader("Cài đặt phân tích video")
            col1, col2 = st.columns(2)
            
            with col1:
                sample_rate = st.slider("Tỷ lệ khung hình phân tích:", 
                                      min_value=1, max_value=30, value=5, 
                                      help="Mỗi bao nhiêu khung hình thì phân tích một lần (1 = phân tích mọi khung hình)")
            
            with col2:
                confidence_threshold = st.slider("Ngưỡng độ tin cậy:", 
                                              min_value=0.5, max_value=1.0, value=0.6, step=0.05,
                                              help="Chỉ hiển thị khuôn mặt có độ tin cậy từ ngưỡng này trở lên")
            
            # Create analysis options
            exp_options = st.expander("Tùy chọn phân tích nâng cao", expanded=False)
            with exp_options:
                col1, col2 = st.columns(2)
                with col1:
                    output_fps = st.slider("Tốc độ video kết quả (FPS):", 
                                         min_value=5, max_value=30, value=15,
                                         help="Số khung hình mỗi giây trong video kết quả")
                
                with col2:
                    max_results = st.slider("Số khuôn mặt tối đa:", 
                                           min_value=1, max_value=10, value=5,
                                           help="Số lượng khuôn mặt tối đa phân tích trong mỗi khung hình")
                
                generate_summary = st.checkbox("Tạo báo cáo thống kê", value=True,
                                             help="Tạo báo cáo thống kê các cá nhân xuất hiện trong video")
                
                save_output = st.checkbox("Lưu video với kết quả nhận dạng", value=True,
                                        help="Tạo video mới có chứa kết quả nhận dạng khuôn mặt")
            
            # Initialize face tracker
            face_tracker = load_face_tracker()
            
            # Analysis button
            analyze_button = st.button("📊 Bắt đầu phân tích", type="primary", use_container_width=True)
            
            if analyze_button:
                # Check if video can be opened
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    st.error("Không thể mở video. Vui lòng kiểm tra lại file video.")
                    try:
                        os.unlink(video_path)  # Clean up temp file
                    except:
                        pass
                else:
                    # Video information
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    original_fps = cap.get(cv2.CAP_PROP_FPS)
                    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    duration = total_frames / original_fps if original_fps > 0 else 0
                    
                    # Display video info
                    st.info(f"Thông tin video: {video_width}x{video_height}, {original_fps:.1f} FPS, {duration:.1f} giây, {total_frames} khung hình")
                    
                    # Setup progress display and storage
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    video_preview = st.empty()
                    
                    # Setup output video writer if needed
                    output_video_path = None
                    video_writer = None
                    if save_output:
                        output_video_path = video_path.replace('.mp4', '_analyzed.mp4')
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        video_writer = cv2.VideoWriter(
                            output_video_path, 
                            fourcc, 
                            output_fps,
                            (video_width, video_height)
                        )
                    
                    # Statistics storage
                    person_appearances = {}  # Store person name -> number of appearances
                    person_first_seen = {}   # Store when the person first appears
                    person_timings = {}      # Store timestamp ranges for each person
                    frame_results = []       # Store detection results for frames
                    
                    # Process video
                    with st.spinner("Đang xử lý video..."):
                        frame_count = 0
                        processed_count = 0
                        
                        while True:
                            ret, frame = cap.read()
                            if not ret:
                                break
                            
                            # Process only every sample_rate frames to save time
                            if frame_count % sample_rate == 0:
                                # Current timestamp
                                timestamp = frame_count / original_fps
                                timestamp_str = f"{int(timestamp // 60):02d}:{int(timestamp % 60):02d}"
                                
                                # Update progress 
                                progress = frame_count / total_frames
                                progress_bar.progress(progress)
                                status_text.text(f"Đang xử lý: {frame_count}/{total_frames} khung hình ({progress*100:.1f}%) - {timestamp_str}")
                                
                                # Detect faces in frame
                                faces, aligned_faces = face_detector.detect(frame)
                                
                                # Update face tracker
                                stable_faces = face_tracker.update(faces, frame_count)
                                
                                # Process only stable faces
                                if len(aligned_faces) > 0 and len(faces) > 0:
                                    # Create a mapping between detected faces and aligned faces
                                    aligned_face_map = {}
                                    for i, face_box in enumerate(faces):
                                        if i < len(aligned_faces):
                                            center_x = int(face_box[0] + face_box[2] / 2)
                                            center_y = int(face_box[1] + face_box[3] / 2)
                                            aligned_face_map[(center_x, center_y)] = aligned_faces[i]
                                    
                                    # Process stable faces
                                    names = []
                                    scores = []
                                    
                                    for face in stable_faces:
                                        if face['is_stable']:
                                            box = face['box']
                                            
                                            # Find closest aligned face
                                            center_x = int(box[0] + box[2] / 2)
                                            center_y = int(box[1] + box[3] / 2)
                                            
                                            # Find closest aligned face within threshold
                                            best_dist = float('inf')
                                            best_face = None
                                            for (x, y), aligned_face in aligned_face_map.items():
                                                dist = abs(x - center_x) + abs(y - center_y)
                                                if dist < best_dist and dist < 50:
                                                    best_dist = dist
                                                    best_face = aligned_face
                                            
                                            # If found, recognize face
                                            if best_face is not None:
                                                name, score = face_recognizer.identify(best_face, confidence_threshold)
                                                names.append(name)
                                                scores.append(score)
                                            else:
                                                names.append("Unknown")
                                                scores.append(0.0)
                                    
                                    # Draw results
                                    from utils.face_utils import draw_stable_results
                                    result_frame = draw_stable_results(frame, stable_faces, names, scores)
                                    
                                    # Add timestamp to frame
                                    cv2.putText(
                                        result_frame, timestamp_str, 
                                        (video_width - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                        0.7, (255, 255, 255), 2
                                    )
                                    
                                    # Update statistics
                                    for i, name in enumerate(names):
                                        if i < len(scores) and scores[i] >= confidence_threshold:
                                            if name != "Unknown":
                                                # Update appearance count
                                                if name in person_appearances:
                                                    person_appearances[name] += 1
                                                else:
                                                    person_appearances[name] = 1
                                                    person_first_seen[name] = timestamp
                                                
                                                # Update timing ranges
                                                if name not in person_timings:
                                                    person_timings[name] = []
                                                
                                                # Check if this is a new timing segment or continuation
                                                if not person_timings[name] or timestamp - person_timings[name][-1][1] > 3.0:
                                                    # New segment (gap > 3 seconds)
                                                    person_timings[name].append([timestamp, timestamp])
                                                else:
                                                    # Update end time of the last segment
                                                    person_timings[name][-1][1] = timestamp
                                    
                                    # Store frame result for summary
                                    frame_results.append({
                                        'frame': frame_count,
                                        'timestamp': timestamp,
                                        'names': [n for i, n in enumerate(names) if i < len(scores) and scores[i] >= confidence_threshold],
                                        'scores': [s for s in scores if s >= confidence_threshold]
                                    })
                                else:
                                    # No faces detected, use original frame
                                    result_frame = frame.copy()
                                    cv2.putText(
                                        result_frame, timestamp_str, 
                                        (video_width - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                        0.7, (255, 255, 255), 2
                                    )
                                
                                # Write to output video if enabled
                                if video_writer is not None:
                                    video_writer.write(result_frame)
                                
                                # Show preview occasionally
                                if processed_count % 10 == 0:
                                    video_preview.image(
                                        cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB),
                                        caption=f"Khung hình {frame_count} - {timestamp_str}",
                                        use_container_width=True
                                    )
                                
                                processed_count += 1
                            
                            frame_count += 1
                            
                            # Check for stop button
                            if st.button("Dừng phân tích", key="stop_video_analysis"):
                                break
                        
                        # Complete progress bar
                        progress_bar.progress(1.0)
                        status_text.text(f"Hoàn thành phân tích video: {processed_count} khung hình đã được xử lý")
                    
                    # Release resources
                    cap.release()
                    if video_writer is not None:
                        video_writer.release()
                    
                    # Display results
                    if generate_summary and person_appearances:
                        st.subheader("📊 Kết quả phân tích")
                        
                        # Display summary of people detected
                        st.markdown("#### Danh sách người được nhận dạng")
                        summary_data = []
                        for name, count in sorted(person_appearances.items(), key=lambda x: x[1], reverse=True):
                            # Calculate percentage of video
                            percentage = (count * sample_rate / total_frames) * 100
                            
                            # Calculate time ranges string
                            time_ranges = []
                            for start, end in person_timings.get(name, []):
                                start_str = f"{int(start // 60):02d}:{int(start % 60):02d}"
                                end_str = f"{int(end // 60):02d}:{int(end % 60):02d}"
                                time_ranges.append(f"{start_str}-{end_str}")
                            
                            time_ranges_str = ", ".join(time_ranges[:3])
                            if len(time_ranges) > 3:
                                time_ranges_str += f" và {len(time_ranges) - 3} khoảng thời gian khác"
                            
                            # First seen
                            first_seen = person_first_seen.get(name, 0)
                            first_seen_str = f"{int(first_seen // 60):02d}:{int(first_seen % 60):02d}"
                            
                            summary_data.append({
                                "Họ tên": name,
                                "Số khung hình xuất hiện": count,
                                "Tỷ lệ xuất hiện": f"{percentage:.1f}%",
                                "Xuất hiện lần đầu": first_seen_str,
                                "Các khoảng thời gian": time_ranges_str
                            })
                        
                        # Display as table
                        st.table(summary_data)
                        
                        # Make a time-based visualization of appearances
                        st.markdown("#### Biểu đồ thời gian xuất hiện")
                        
                        # Create horizontal timeline visualization
                        timeline_height = 80
                        person_height = 20
                        padding = 5
                        text_width = 150
                        
                        # Calculate timeline width based on duration
                        seconds_width = 4  # pixels per second
                        timeline_width = int(max(800, duration * seconds_width))
                        
                        # Create timeline image
                        timeline_image = np.ones((len(person_timings) * (person_height + padding) + padding, 
                                                timeline_width + text_width, 3), dtype=np.uint8) * 255
                        
                        # Draw timeline for each person
                        for i, (name, time_ranges) in enumerate(person_timings.items()):
                            # Calculate y position
                            y_pos = padding + i * (person_height + padding)
                            
                            # Draw name
                            cv2.putText(
                                timeline_image, name, 
                                (5, y_pos + person_height - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1
                            )
                            
                            # Draw timeline base
                            cv2.line(
                                timeline_image,
                                (text_width, y_pos + person_height // 2),
                                (text_width + timeline_width, y_pos + person_height // 2),
                                (200, 200, 200), 1
                            )
                            
                            # Draw appearance blocks
                            for start, end in time_ranges:
                                start_pos = text_width + int(start * seconds_width)
                                end_pos = text_width + int(end * seconds_width)
                                
                                cv2.rectangle(
                                    timeline_image,
                                    (start_pos, y_pos),
                                    (end_pos, y_pos + person_height),
                                    (0, 150, 0), -1
                                )
                            
                        # Draw time markers
                        marker_interval = 60  # 1 minute
                        for t in range(0, int(duration) + marker_interval, marker_interval):
                            x_pos = text_width + int(t * seconds_width)
                            
                            # Draw vertical line
                            cv2.line(
                                timeline_image,
                                (x_pos, 0),
                                (x_pos, timeline_image.shape[0]),
                                (180, 180, 180), 1
                            )
                            
                            # Draw time label
                            time_label = f"{t//60}:{t%60:02d}"
                            cv2.putText(
                                timeline_image, time_label,
                                (x_pos - 20, timeline_image.shape[0] - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1
                            )
                        
                        # Display timeline
                        st.image(timeline_image, use_container_width=True)
                    
                    # If video was saved, provide download link
                    if save_output and output_video_path and os.path.exists(output_video_path):
                        # Read the video file
                        with open(output_video_path, 'rb') as file:
                            video_bytes = file.read()
                        
                        # Create download button
                        st.download_button(
                            label="⬇️ Tải xuống video kết quả",
                            data=video_bytes,
                            file_name=f"video_analyzed.mp4",
                            mime="video/mp4"
                        )
                    
                    # Clean up temporary files
                    try:
                        if os.path.exists(video_path):
                            os.unlink(video_path)
                        if output_video_path and os.path.exists(output_video_path):
                            # Don't delete right away, as it might be used for download
                            pass
                    except:
                        pass
                    
                    st.success("✅ Phân tích video hoàn tất!")