import numpy as np
import streamlit as st
import cv2
import os
import unicodedata
import re
import time
import threading
from datetime import datetime

def name_to_folder_name(name):
    """Convert Vietnamese name to folder name format"""
    # Convert to ASCII and lowercase
    name = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('ascii')
    name = name.lower()
    # Replace spaces with underscores
    name = re.sub(r'[^a-z0-9]+', '_', name).strip('_')
    return name

@st.cache_resource
def load_face_detector():
    from utils.face_utils import FaceDetector
    return FaceDetector("models/face_detection_yunet_2023mar.onnx")

@st.cache_resource
def load_face_recognizer():
    from utils.face_utils import FaceRecognizer
    return FaceRecognizer("models/face_recognition_sface_2021dec.onnx", "data/db_embeddings.pkl")

def show():
    # Thêm phần giới thiệu
    with st.expander("🔍 Giới thiệu về đăng ký khuôn mặt", expanded=False):
        st.markdown("""
        ### Giới thiệu về đăng ký khuôn mặt
        
        Tính năng đăng ký khuôn mặt mới là một phần quan trọng trong hệ thống nhận dạng khuôn mặt, cho phép hệ thống học và nhận dạng khuôn mặt của người dùng mới.
        
        #### Quy trình đăng ký khuôn mặt
        
        Quy trình đăng ký khuôn mặt bao gồm các bước chính sau:
        
        1. **Thu thập dữ liệu**: Chụp nhiều ảnh khuôn mặt từ các góc và điều kiện ánh sáng khác nhau
        2. **Phát hiện khuôn mặt**: Sử dụng YuNet để phát hiện và cắt vùng khuôn mặt từ ảnh
        3. **Căn chỉnh khuôn mặt**: Căn chỉnh khuôn mặt để chuẩn hóa vị trí của các đặc điểm
        4. **Trích xuất đặc trưng**: Sử dụng SFace để chuyển đổi ảnh khuôn mặt thành vector embedding 128 chiều
        5. **Lưu trữ database**: Lưu trữ vector đặc trưng cùng với nhãn (tên người) vào database
        
        #### Công nghệ sử dụng
        
        - **YuNet** để phát hiện khuôn mặt với độ chính xác cao, ngay cả trong điều kiện ánh sáng khác nhau
        - **SFace** để trích xuất đặc trưng khuôn mặt thành vector 128 chiều, tạo "chữ ký sinh trắc học" duy nhất
        - **Phương pháp chuẩn hóa** để đảm bảo các vector có cùng độ dài và dễ dàng so sánh
        - **Lưu trữ hiệu quả** sử dụng Pickle để lưu trữ database vector đặc trưng
        
        #### Tầm quan trọng của số lượng mẫu
        
        Hệ thống thu thập 100 mẫu khuôn mặt cho mỗi người để:
        - Tăng độ chính xác trong nhận dạng
        - Đảm bảo nhận dạng được trong nhiều điều kiện khác nhau
        - Giảm thiểu lỗi nhận dạng sai
        
        Ứng dụng này cung cấp hai phương pháp đăng ký khuôn mặt: tự động và thủ công, giúp người dùng có thể lựa chọn phương pháp phù hợp nhất với nhu cầu của họ.
        """)
            
    # Thêm phần hướng dẫn
    with st.expander("📋 Hướng dẫn sử dụng", expanded=False):
        st.markdown("""
        ### Hướng dẫn sử dụng
        
        #### 1. Chế độ tự động (Khuyên dùng)
        
        Chế độ này tự động chụp và xử lý ảnh khuôn mặt:
        
        1. **Chuẩn bị**:
           - Đảm bảo camera hoạt động tốt và được kết nối
           - Đặt camera ở nơi có đủ ánh sáng
           - Giữ khuôn mặt trong khung hình từ 0.5-1m
        
        2. **Nhập thông tin**:
           - Nhập họ tên đầy đủ (sẽ được dùng để nhận dạng sau này)
           - Điều chỉnh số lượng ảnh mẫu nếu cần (mặc định: 100)
        
        3. **Quá trình chụp**:
           - Nhấn "🚀 Bắt đầu" để bắt đầu quá trình
           - Di chuyển đầu nhẹ nhàng để có nhiều góc khác nhau
           - Có thể nhấn "⏸️ Tạm dừng" nếu cần nghỉ giữa chừng
           - Nhấn "⏹️ Dừng" để kết thúc quá trình sớm
        
        4. **Cập nhật database**:
           - Khi đã đủ số lượng ảnh, nhấn "Cập nhật Database"
           - Đợi quá trình xử lý và tạo vector đặc trưng hoàn tất
        
        #### 2. Chế độ thủ công
        
        Phù hợp khi bạn muốn kiểm soát từng ảnh được chụp:
        
        1. **Thiết lập**:
           - Nhập họ tên đầy đủ
           - Điều chỉnh số lượng ảnh mẫu
           - Nhấn "🚀 Bắt đầu" để bắt đầu
        
        2. **Chụp ảnh**:
           - Nhấn "Space" hoặc nút chụp để chụp ảnh
           - Kiểm tra khuôn mặt đã được phát hiện đúng chưa (khung xanh)
           - Nhấn "💾 Lưu ảnh" để lưu mẫu
           - Nhấn "🔄 Clear Photo" để chụp lại
        
        #### Mẹo để có kết quả tốt nhất
        
        - **Ánh sáng**: Đảm bảo khuôn mặt được chiếu sáng đầy đủ và đều
        - **Biểu cảm**: Thay đổi biểu cảm nhẹ (mỉm cười, nghiêm túc) để tăng độ đa dạng
        - **Góc nhìn**: Di chuyển đầu nhẹ nhàng sang trái, phải, lên, xuống
        - **Phụ kiện**: Thử đeo/bỏ kính, thay đổi kiểu tóc nếu có thể
        - **Tránh chuyển động nhanh**: Di chuyển từ từ để tránh ảnh bị mờ
        
        #### Quản lý database
        
        - Sử dụng chức năng "Xây dựng lại Database hoàn toàn" nếu muốn làm mới hệ thống
        - "Kiểm tra Database hiện tại" để xem danh sách người đã đăng ký và số lượng mẫu
        - Restart ứng dụng sau khi cập nhật database để áp dụng thay đổi
        """)
            
    # Choose capture mode
    tab1, tab2 = st.tabs(["🎥 Chế độ tự động", "👆 Chế độ thủ công"])
    
    with tab1:
        show_interactive_capture_ui()
    
    with tab2:
        show_manual_capture_ui()
    
    # Additional database management
    st.markdown("---")
    st.markdown("### Quản lý Database")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Xây dựng lại Database hoàn toàn", key="main_rebuild_db_button"):
            with st.spinner("Đang xây dựng database..."):
                detector = load_face_detector()
                recognizer = load_face_recognizer()
                output_dir = "data/faces"
                from utils.face_utils import build_face_database
                total_count = build_face_database(detector, recognizer, output_dir, "data/db_embeddings.pkl")
                st.success(f"✅ Database đã được xây dựng với {total_count} khuôn mặt!")
                st.info("Vui lòng restart ứng dụng để áp dụng thay đổi.")
    
    with col2:
        if st.button("Kiểm tra Database hiện tại", key="main_check_db_button"):
            db_path = "data/db_embeddings.pkl"
            if os.path.exists(db_path):
                try:
                    import pickle
                    with open(db_path, 'rb') as f:
                        database = pickle.load(f)
                    
                    st.success("Database tồn tại và hợp lệ")
                    st.write(f"Số người: {len(database)}")
                    for name, features in database.items():
                        st.write(f"- {name}: {len(features)} ảnh")
                except Exception as e:
                    st.error(f"Lỗi khi đọc database: {e}")
            else:
                st.warning("Database không tồn tại. Vui lòng xây dựng database mới.")

def show_interactive_capture_ui():
    """Create an interactive capture UI without frequent reloads"""
    
    st.markdown("### 🎥 Chế độ chụp tự động (Interactive)")
    
    # Initialize session state
    if 'capture_running' not in st.session_state:
        st.session_state.capture_running = False
    if 'current_image_count' not in st.session_state:
        st.session_state.current_image_count = 0
    if 'person_name' not in st.session_state:
        st.session_state.person_name = ""
    if 'capturing_status' not in st.session_state:
        st.session_state.capturing_status = "Ready"  # Ready, Capturing, Paused
    if 'force_update' not in st.session_state:
        st.session_state.force_update = False
    if 'num_samples' not in st.session_state:
        st.session_state.num_samples = 100
    if 'folder_name' not in st.session_state:
        st.session_state.folder_name = ""
    
    # Input form
    col1, col2 = st.columns([3, 1])
    with col1:
        person_name = st.text_input("Họ và tên", value=st.session_state.person_name, key="auto_person_name")
    with col2:
        num_samples = st.number_input("Số ảnh mẫu", min_value=50, max_value=200, value=100, key="auto_num_samples")
    
    # Control buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        start_button = st.button("🚀 Bắt đầu", disabled=st.session_state.capture_running, key="auto_start_button")
    with col2:
        pause_resume = st.button(
            "⏸️ Tạm dừng" if st.session_state.capturing_status != "Paused" else "▶️ Tiếp tục",
            disabled=not st.session_state.capture_running,
            key="auto_pause_resume_button"
        )
    with col3:
        stop_button = st.button("⏹️ Dừng", disabled=not st.session_state.capture_running, key="auto_stop_button")
    
    # Start capture
    if start_button and person_name:
        st.session_state.capture_running = True
        st.session_state.current_image_count = 0
        st.session_state.person_name = person_name
        st.session_state.num_samples = num_samples
        st.session_state.folder_name = name_to_folder_name(person_name)
        st.session_state.capturing_status = "Capturing"
        st.session_state.force_update = True
        st.rerun()
    elif start_button and not person_name:
        st.error("Vui lòng nhập họ tên!")
    
    # Pause/Resume capture
    if pause_resume:
        if st.session_state.capturing_status == "Capturing":
            st.session_state.capturing_status = "Paused"
        else:
            st.session_state.capturing_status = "Capturing"
        # Don't rerun for pause/resume to avoid camera reload
    
    # Stop capture
    if stop_button:
        st.session_state.capture_running = False
        st.session_state.capturing_status = "Ready"
        st.session_state.force_update = True
        st.rerun()
    
    # Progress display (create placeholders)
    progress_placeholder = st.empty()
    status_placeholder = st.empty()
    count_placeholder = st.empty()
    video_placeholder = st.empty()
    
    # Update progress displays
    if st.session_state.capture_running:
        progress_placeholder.progress(st.session_state.current_image_count / st.session_state.num_samples)
        status_placeholder.write(f"**Trạng thái:** {st.session_state.capturing_status}")
        count_placeholder.write(f"**Tiến trình:** {st.session_state.current_image_count}/{st.session_state.num_samples}")
        
        # Create directories
        output_dir = "data/faces"
        os.makedirs(output_dir, exist_ok=True)
        person_dir = os.path.join(output_dir, st.session_state.folder_name)
        os.makedirs(person_dir, exist_ok=True)
        
        # Start capture process
        if st.session_state.current_image_count < st.session_state.num_samples:
            detector = load_face_detector()
            
            # Open camera only once
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Không thể mở camera!")
                st.session_state.capture_running = False
            else:
                # Camera capture loop
                frame_count = 0
                delay_frames = 3
                last_capture_time = time.time()
                min_capture_interval = 0.1  # Minimum 0.1 second between captures
                
                while st.session_state.capture_running and st.session_state.current_image_count < st.session_state.num_samples:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Không thể đọc từ camera!")
                        break
                    
                    # Flip frame for mirror effect
                    frame = cv2.flip(frame, 1)
                    
                    # Detect faces
                    faces, aligned_faces = detector.detect(frame)
                    
                    # Draw on frame
                    display_frame = frame.copy()
                    for face in faces:
                        x, y, w, h = map(int, face[:4])
                        confidence = face[4]
                        
                        # Draw face rectangle
                        cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(display_frame, f"Face: {confidence:.2f}", (x, y-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                    # Add status text
                    status_color = (0, 255, 0) if st.session_state.capturing_status == "Capturing" else (0, 0, 255)
                    status_text = st.session_state.capturing_status
                    cv2.putText(display_frame, status_text, (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                    
                    # Add count text
                    cv2.putText(display_frame, f"Captured: {st.session_state.current_image_count}/{st.session_state.num_samples}", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Display frame
                    video_placeholder.image(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                    
                    # Update progress displays without rerunning
                    current_time = time.time()
                    progress_placeholder.progress(st.session_state.current_image_count / st.session_state.num_samples)
                    status_placeholder.write(f"**Trạng thái:** {st.session_state.capturing_status}")
                    count_placeholder.write(f"**Tiến trình:** {st.session_state.current_image_count}/{st.session_state.num_samples}")
                    
                    # Save face if capturing and conditions met
                    if (st.session_state.capturing_status == "Capturing" and 
                        len(aligned_faces) > 0 and 
                        current_time - last_capture_time >= min_capture_interval):
                        
                        # Save the first face detected
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        filename = os.path.join(person_dir, f"{st.session_state.current_image_count:04d}_{timestamp}.jpg")
                        cv2.imwrite(filename, aligned_faces[0])
                        
                        # Update count and time
                        st.session_state.current_image_count += 1
                        last_capture_time = current_time
                        
                        # Update display without rerunning
                        progress_placeholder.progress(st.session_state.current_image_count / st.session_state.num_samples)
                        count_placeholder.write(f"**Tiến trình:** {st.session_state.current_image_count}/{st.session_state.num_samples}")
                    
                    # Minimal delay to allow UI updates
                    time.sleep(0.01)
                    
                    # Break if user stopped or completed
                    if not st.session_state.capture_running or st.session_state.current_image_count >= st.session_state.num_samples:
                        break
                
                cap.release()
    
    # Completion check
    if st.session_state.current_image_count >= st.session_state.num_samples:
        st.success(f"🎉 Hoàn thành! Đã chụp {st.session_state.current_image_count} ảnh")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Cập nhật Database", key="auto_update_db_button"):
                with st.spinner("Đang xây dựng database..."):
                    detector = load_face_detector()
                    recognizer = load_face_recognizer()
                    from utils.face_utils import build_face_database
                    total_count = build_face_database(detector, recognizer, output_dir, "data/db_embeddings.pkl")
                    st.success(f"✅ Database đã được cập nhật với {total_count} khuôn mặt!")
                    st.info("Vui lòng restart ứng dụng để áp dụng thay đổi.")
        
        with col2:
            if st.button("Đăng ký người khác", key="auto_register_another_button"):
                st.session_state.capture_running = False
                st.session_state.current_image_count = 0
                st.session_state.person_name = ""
                st.session_state.capturing_status = "Ready"
                st.session_state.force_update = True
                st.rerun()

def show_manual_capture_ui():
    """Manual capture interface with real-time face detection"""
    st.markdown("### 👆 Chế độ chụp thủ công")
    
    # Initialize session state
    if 'manual_capture_running' not in st.session_state:
        st.session_state.manual_capture_running = False
    if 'manual_current_image_count' not in st.session_state:
        st.session_state.manual_current_image_count = 0
    if 'manual_folder_name' not in st.session_state:
        st.session_state.manual_folder_name = ""
    if 'manual_captured_image' not in st.session_state:
        st.session_state.manual_captured_image = None
    
    # Input form
    col1, col2 = st.columns([3, 1])
    with col1:
        person_name = st.text_input("Họ và tên", key="manual_person_name")
    with col2:
        num_samples = st.number_input("Số ảnh mẫu", min_value=50, max_value=200, value=100, key="manual_num_samples")
    
    # Control buttons
    col1, col2 = st.columns(2)
    
    with col1:
        start_button = st.button("🚀 Bắt đầu", disabled=st.session_state.manual_capture_running, key="manual_start_button")
    with col2:
        stop_button = st.button("⏹️ Dừng", disabled=not st.session_state.manual_capture_running, key="manual_stop_button")
    
    # Instructions
    if st.session_state.manual_capture_running:
        st.info("**Nhấn SPACE hoặc 'Take Photo' để chụp ảnh | Nhấn 'Clear Photo' để xóa và chụp ảnh mới**")
    
    # Start capture
    if start_button and person_name:
        st.session_state.manual_capture_running = True
        st.session_state.manual_current_image_count = 0
        st.session_state.manual_folder_name = name_to_folder_name(person_name)
        st.session_state.manual_captured_image = None
        st.rerun()
    elif start_button and not person_name:
        st.error("Vui lòng nhập họ tên!")
    
    # Stop capture
    if stop_button:
        st.session_state.manual_