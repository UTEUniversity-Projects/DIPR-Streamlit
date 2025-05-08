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

@st.cache_resource(ttl=10)  # Time to live: chỉ 10 giây để tải lại nhanh hơn
def load_face_recognizer():
    db_path = "data/db_embeddings.pkl"
    # Thêm mtime làm tham số để Streamlit biết khi nào cần tải lại
    db_mtime = os.path.getmtime(db_path) if os.path.exists(db_path) else 0
    from utils.face_utils import FaceRecognizer
    recognizer = FaceRecognizer("models/face_recognition_sface_2021dec.onnx", db_path)
    print(f"Loaded face recognizer with database timestamp: {db_mtime}")
    return recognizer

def draw_results(frame: np.ndarray, faces: np.ndarray, names: List[str], 
                scores: List[float]) -> np.ndarray:
    """
    Draw detection and recognition results on the frame - simplified version
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
        # Skip if index out of range
        if i >= len(names) or i >= len(scores):
            continue
            
        x, y, w, h = map(int, face[:4])
        
        # Get name and score
        name = names[i]
        score = scores[i]
        
        # Choose color based on recognition status
        if name == "Unknown":
            color = (0, 0, 255)  # Red for unknown
            text_color = (255, 255, 255)  # White text
        else:
            color = (0, 255, 0)  # Green for known
            text_color = (255, 255, 255)  # White text
        
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
        
        **Các tính năng chính:**
        - Nhận dạng khuôn mặt từ ảnh tĩnh (tải lên hoặc chụp từ webcam)
        - Nhận dạng khuôn mặt từ video tải lên (nhiều định dạng như MP4, AVI, MOV, v.v.)
        - Phát hiện và nhận dạng khuôn mặt từ webcam theo thời gian thực
        """)
            
    with st.expander("📋 Hướng dẫn sử dụng", expanded=False):
        st.markdown("""
        ### Hướng dẫn sử dụng
        
        #### 1. Chế độ ảnh tĩnh
        - **Upload ảnh**: Tải lên ảnh chứa khuôn mặt cần nhận dạng
        - **Chụp từ webcam**: Chụp ảnh trực tiếp từ webcam để nhận dạng
        
        #### 2. Chế độ video tải lên
        - **Upload video**: Tải lên video có định dạng MP4, AVI, MOV, MKV, v.v.
        - **Điều chỉnh tốc độ xử lý**: Chọn tốc độ xử lý (cao hơn = nhanh hơn)
        - **Điều chỉnh độ nhạy**: Thay đổi ngưỡng xác định cho kết quả tốt hơn
        - **Hiển thị video**: Xem video với khuôn mặt được đánh dấu trực tiếp
        
        #### 3. Chế độ video trực tiếp
        - **Bắt đầu**: Mở camera và bắt đầu phát hiện khuôn mặt
        - **Dừng**: Dừng quá trình nhận dạng và đóng camera
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
                        # Sắp xếp khuôn mặt theo kích thước (lớn -> nhỏ)
                        face_sizes = [f[2] * f[3] for f in faces]  # width * height
                        sorted_indices = np.argsort(face_sizes)[::-1]  # Giảm dần
                        
                        sorted_faces = faces[sorted_indices]
                        sorted_aligned_faces = [aligned_faces[i] for i in sorted_indices if i < len(aligned_faces)]
                        
                        names = []
                        scores = []
                        
                        # Nhận diện khuôn mặt (limit to 5)
                        for face_img in sorted_aligned_faces[:5]:
                            # Chỉ sử dụng ảnh gốc, loại bỏ phương pháp flip
                            name, score = face_recognizer.identify(face_img, threshold=0.45)
                            names.append(name)
                            scores.append(score)
                        
                        # Draw results - sử dụng hàm đơn giản hóa
                        result_img = draw_results(img, sorted_faces[:5], names, scores)
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
                        # Sắp xếp khuôn mặt theo kích thước (lớn -> nhỏ)
                        face_sizes = [f[2] * f[3] for f in faces]  # width * height
                        sorted_indices = np.argsort(face_sizes)[::-1]  # Giảm dần
                        
                        sorted_faces = faces[sorted_indices]
                        sorted_aligned_faces = [aligned_faces[i] for i in sorted_indices if i < len(aligned_faces)]
                        
                        names = []
                        scores = []
                        
                        # Nhận diện khuôn mặt (limit to 5)
                        for face_img in sorted_aligned_faces[:5]:
                            # Chỉ sử dụng ảnh gốc, loại bỏ phương pháp flip
                            name, score = face_recognizer.identify(face_img, threshold=0.45)
                            names.append(name)
                            scores.append(score)
                        
                        # Draw results - sử dụng hàm đơn giản hóa
                        result_img = draw_results(img, sorted_faces[:5], names, scores)
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
        Video sẽ được xử lý trực tiếp với kết quả nhận dạng hiển thị ngay trên video!
        """)
        
        # Upload video file with clearer instructions
        video_file = st.file_uploader(
            "Tải lên video chứa khuôn mặt cần nhận dạng", 
            type=["mp4", "mov", "avi", "mkv", "wmv"],
            help="Hỗ trợ các định dạng: MP4, MOV, AVI, MKV, WMV. Nên sử dụng video MP4 để có hiệu suất tốt nhất."
        )
        
        if video_file is not None:
            # Sử dụng hàm xử lý video đã cải tiến
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
            process_rate = st.slider("Tốc độ xử lý:", 1, 6, 2, 1,
                                    help="Số frame bỏ qua giữa các lần xử lý (cao hơn = nhanh hơn)")
        
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
                
                # FPS counter
                fps_start_time = time.time()
                fps_counter = 0
                current_fps = 0
                frame_num = 0
                
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
                        
                        # Nếu phát hiện các khuôn mặt, sắp xếp chúng theo kích thước
                        if len(faces) > 0:
                            face_sizes = [f[2] * f[3] for f in faces]  # width * height
                            sorted_indices = np.argsort(face_sizes)[::-1]  # Giảm dần
                            
                            sorted_faces = faces[sorted_indices]
                            sorted_aligned_faces = [aligned_faces[i] for i in sorted_indices if i < len(aligned_faces)]
                            
                            # Nhận diện khuôn mặt
                            names = []
                            scores = []
                            
                            for face_img in sorted_aligned_faces[:5]:  # Chỉ xử lý tối đa 5 khuôn mặt
                                # Chỉ sử dụng ảnh gốc, loại bỏ phương pháp flip
                                name, score = face_recognizer.identify(face_img, threshold=0.45)
                                names.append(name)
                                scores.append(score)
                            
                            # Vẽ kết quả
                            display_frame = draw_results(frame, sorted_faces, names, scores)
                            
                            # Display recognition info
                            if len(names) > 0:
                                info_text = "### Khuôn mặt được nhận diện:\n\n"
                                
                                for i, (name, score) in enumerate(zip(names, scores)):
                                    status = "✅ Đã nhận dạng" if name != "Unknown" else "❌ Không nhận dạng được"
                                    info_text += f"**Người {i+1}:** {name} ({score:.2f}) - {status}\n\n"
                                
                                info_placeholder.markdown(info_text)
                        else:
                            display_frame = frame
                            if frame_num % (process_rate * 5) == 0:
                                info_placeholder.markdown("*Không phát hiện khuôn mặt nào*")
                    else:
                        display_frame = frame
                    
                    frame_num += 1
                    
                    # Calculate FPS
                    fps_counter += 1
                    if time.time() - fps_start_time >= 1.0:
                        current_fps = fps_counter
                        fps_counter = 0
                        fps_start_time = time.time()
                    
                    # Add FPS text to frame
                    cv2.putText(display_frame, f"FPS: {current_fps}", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # Display frame
                    video_placeholder.image(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB), 
                                          channels="RGB", use_container_width=True)
                    
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
            - Hiển thị khuôn mặt trong thời gian thực
            - Tự động nhận dạng khuôn mặt trong thời gian thực
            - Hiển thị FPS để theo dõi hiệu suất
            
            **Tùy chọn hiệu suất:**
            - Độ phân giải thấp = nhanh hơn
            - Tốc độ xử lý cao = xử lý nhiều frame hơn/giây
            
            Nhấn **Bắt đầu** để mở camera.
            """)
            
def process_video_realtime(video_file, face_detector, face_recognizer):
    """
    Cải tiến xử lý video theo thời gian thực - trực tiếp hiển thị kết quả nhận diện lên video
    """
    import tempfile
    import os
    import cv2
    import numpy as np
    import time
    from datetime import timedelta
    import streamlit as st
    
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
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0
    
    # Hiển thị thông tin video
    st.info(f"Thông tin video: {width}x{height}, {fps:.1f} FPS, {frame_count} frames, thời lượng: {timedelta(seconds=duration)}")
    
    # Điều chỉnh độ nhạy và tốc độ xử lý
    col1, col2 = st.columns(2)
    with col1:
        confidence_threshold = st.slider(
            "Ngưỡng nhận diện:", 
            min_value=0.4, 
            max_value=0.95, 
            value=0.45, 
            step=0.05,
            help="Điều chỉnh độ nhạy khi nhận diện (cao hơn = ít nhận diện sai hơn)"
        )
    with col2:
        processing_speed = st.slider(
            "Tốc độ xử lý:", 
            min_value=1, 
            max_value=10, 
            value=3, 
            step=1,
            help="Tốc độ xử lý (cao hơn = nhanh hơn nhưng bỏ qua nhiều frame)"
        )
    
    # Thiết lập video player và controls
    video_placeholder = st.empty()
    progress_bar = st.progress(0.0)
    info_display = st.empty()
    
    # Controls
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        start_button = st.button("▶️ Bắt đầu", type="primary", use_container_width=True)
    with col2:
        pause_button = st.button("⏸️ Tạm dừng", use_container_width=True)
    with col3:
        stop_button = st.button("⏹️ Dừng", use_container_width=True)
    
    # Thông tin nhận diện
    stats = {
        'people_detected': set(),
        'people_frames': {},
        'total_faces': 0,
        'identified_faces': 0,
        'unknown_faces': 0
    }
    
    # Theo dõi trạng thái
    if 'video_playing' not in st.session_state:
        st.session_state.video_playing = False
    if 'video_paused' not in st.session_state:
        st.session_state.video_paused = False
    if 'current_frame' not in st.session_state:
        st.session_state.current_frame = 0
        
    # Xử lý nút bấm
    if start_button:
        st.session_state.video_playing = True
        st.session_state.video_paused = False
        if st.session_state.current_frame >= frame_count:
            st.session_state.current_frame = 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    if pause_button:
        st.session_state.video_paused = not st.session_state.video_paused
    
    if stop_button:
        st.session_state.video_playing = False
        st.session_state.video_paused = False
        st.session_state.current_frame = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # Xử lý phát video
    if st.session_state.video_playing:
        # Đặt vị trí frame nếu cần
        if st.session_state.current_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.current_frame)
        
        # Biến theo dõi tốc độ FPS
        start_time = time.time()
        processed_frames = 0
        display_fps = 0
        
        # Tạo file video output tạm thời
        temp_output_path = temp_video_path + "_output.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = None  # Sẽ được khởi tạo sau
        
        # Chuẩn bị thông tin hiện thị
        info_text = ""
        
        try:
            while st.session_state.video_playing and st.session_state.current_frame < frame_count:
                # Kiểm tra tạm dừng
                if st.session_state.video_paused:
                    time.sleep(0.1)  # Tạm dừng ngắn để tránh treo
                    continue
                
                # Đọc frame
                ret, frame = cap.read()
                if not ret:
                    break
                
                st.session_state.current_frame += 1
                
                # Chỉ xử lý một số frame (bỏ qua frame để tăng tốc)
                if st.session_state.current_frame % processing_speed != 0:
                    continue
                
                # Phát hiện khuôn mặt
                faces, aligned_faces = face_detector.detect(frame)
                
                # Cập nhật thống kê
                if len(faces) > 0:
                    stats['total_faces'] += len(faces)
                    
                    # Sắp xếp khuôn mặt theo kích thước (lớn -> nhỏ)
                    face_sizes = [f[2] * f[3] for f in faces]  # width * height
                    sorted_indices = np.argsort(face_sizes)[::-1]  # Giảm dần
                    
                    sorted_faces = faces[sorted_indices]
                    sorted_aligned_faces = [aligned_faces[i] for i in sorted_indices if i < len(aligned_faces)]
                    
                    names = []
                    scores = []
                    
                    # Nhận diện khuôn mặt (giới hạn 5 khuôn mặt)
                    for face_img in sorted_aligned_faces[:5]:
                        name, score = face_recognizer.identify(face_img, threshold=confidence_threshold)
                        
                        # Cập nhật thống kê
                        if name != "Unknown":
                            stats['identified_faces'] += 1
                            stats['people_detected'].add(name)
                            stats['people_frames'][name] = stats['people_frames'].get(name, 0) + 1
                        else:
                            stats['unknown_faces'] += 1
                            
                        names.append(name)
                        scores.append(score)
                    
                    # Vẽ kết quả
                    result_frame = draw_results(frame, sorted_faces[:5], names, scores)
                else:
                    result_frame = frame
                
                # Tính FPS
                processed_frames += 1
                elapsed_time = time.time() - start_time
                if elapsed_time >= 1.0:  # Cập nhật FPS mỗi giây
                    display_fps = processed_frames / elapsed_time
                    processed_frames = 0
                    start_time = time.time()
                
                # Thêm thông tin FPS vào frame
                cv2.putText(
                    result_frame,
                    f"FPS: {display_fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2
                )
                
                # Thêm thông tin vị trí frame
                position_percent = st.session_state.current_frame / frame_count
                cv2.putText(
                    result_frame,
                    f"Frame: {st.session_state.current_frame}/{frame_count}",
                    (10, height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )
                
                # Cập nhật thanh tiến trình
                progress_bar.progress(position_percent)
                
                # Hiển thị frame đã xử lý
                video_placeholder.image(
                    cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB),
                    channels="RGB",
                    use_container_width=True
                )
                
                # Khởi tạo output writer nếu cần
                if out is None:
                    out = cv2.VideoWriter(
                        temp_output_path,
                        fourcc,
                        fps,
                        (width, height)
                    )
                
                # Lưu frame vào video output
                out.write(result_frame)
                
                # Hiển thị thông tin nhận diện hiện tại
                if len(stats['people_detected']) > 0:
                    info_text = f"**Nhận diện được:** {len(stats['people_detected'])} người"
                    info_text += f" | **Khuôn mặt đã xử lý:** {stats['total_faces']}"
                    info_text += f" | **Đã nhận diện:** {stats['identified_faces']}"
                    info_text += f" | **Không nhận diện được:** {stats['unknown_faces']}"
                    
                    # Top 3 người xuất hiện nhiều nhất
                    if stats['people_frames']:
                        sorted_people = sorted(stats['people_frames'].items(), key=lambda x: x[1], reverse=True)
                        top_people = sorted_people[:3]
                        info_text += "\n\n**Người xuất hiện nhiều nhất:**"
                        for name, count in top_people:
                            info_text += f" {name} ({count} frames),"
                        info_text = info_text.rstrip(",")
                
                info_display.markdown(info_text)
                
                # Kiểm tra trạng thái
                if not st.session_state.video_playing or st.session_state.current_frame >= frame_count:
                    break
                
            # Đóng video writer
            if out is not None:
                out.release()
            
            # Hiển thị thông báo kết thúc nếu đã xem hết video
            if st.session_state.current_frame >= frame_count:
                st.success("Đã hoàn thành phát video!")
                st.session_state.video_playing = False
                
                # Hiển thị kết quả cuối cùng
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
                                frame_percent = frames / frame_count * 100
                                st.metric(
                                    label=person,
                                    value=f"{frames} frames",
                                    delta=f"{frame_percent:.1f}%"
                                )
                else:
                    st.warning("Không nhận diện được người nào trong video.")
                
                # Tải về video đã xử lý
                if os.path.exists(temp_output_path):
                    with open(temp_output_path, 'rb') as f:
                        video_bytes = f.read()
                        
                    st.download_button(
                        label="Tải xuống video đã xử lý",
                        data=video_bytes,
                        file_name="video_with_face_recognition.mp4",
                        mime="video/mp4"
                    )
        
        except Exception as e:
            st.error(f"Lỗi khi xử lý video: {str(e)}")
        
        finally:
            # Giải phóng resources
            cap.release()
            if out is not None:
                out.release()
    
    else:
        # Hiển thị hướng dẫn khi không phát
        video_placeholder.markdown("""
        ## 🎬 Xử lý video nhận diện khuôn mặt
        
        Tải lên video của bạn và nhấn **▶️ Bắt đầu** để bắt đầu xử lý.
        
        Ứng dụng sẽ phát hiện và nhận diện khuôn mặt trong video, hiển thị trực tiếp kết quả trên màn hình.
        
        **Tính năng:**
        - Hiển thị video với nhận diện khuôn mặt theo thời gian thực
        - Theo dõi số lượng khuôn mặt được nhận diện
        - Điều chỉnh độ nhạy và tốc độ xử lý
        - Tải xuống video đã xử lý
        """)
    
    # Dọn dẹp
    try:
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)
    except:
        pass