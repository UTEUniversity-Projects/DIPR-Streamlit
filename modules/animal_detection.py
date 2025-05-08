import streamlit as st
import cv2
import numpy as np
import os

def show():
    # Add introduction section
    with st.expander("🔍 Giới thiệu về nhận dạng động vật", expanded=False):
        st.markdown("""
        ### Giới thiệu về nhận dạng động vật
        
        Tính năng nhận dạng động vật sử dụng mô hình YOLOv8, một trong những mô hình phát hiện đối tượng hiệu quả nhất hiện nay. Mô hình được đào tạo để nhận diện nhiều loài động vật khác nhau trong ảnh.
        
        #### Kiến trúc YOLOv8
        
        YOLOv8 (You Only Look Once version 8) là phiên bản mới nhất của dòng mô hình YOLO với nhiều cải tiến:
        
        - **Backbone**: Hiệu quả hơn với CSPDarknet được tối ưu hóa
        - **Neck**: Cải tiến FPN (Feature Pyramid Network) giúp nhận diện đối tượng ở nhiều kích thước
        - **Head**: Đầu ra anchor-free giúp tăng độ chính xác
        - **Loss Function**: Hàm mất mát được cải tiến giúp phân loại và xác định vị trí tốt hơn
        
        #### Khả năng nhận dạng
        
        Mô hình có thể nhận diện nhiều loài động vật phổ biến như:
        - Chó (Dog)
        - Mèo (Cat)
        - Chim (Bird)
        - Ngựa (Horse)
        - Bò (Cow)
        - Voi (Elephant)
        - Gấu (Bear)
        - Ngựa vằn (Zebra)
        - Hươu cao cổ (Giraffe)
        - Hổ (Tiger)
        
        #### Ứng dụng thực tế
        
        Nhận dạng động vật có nhiều ứng dụng quan trọng:
        - Theo dõi và bảo tồn động vật hoang dã
        - Nghiên cứu hành vi động vật
        - Hệ thống giám sát tự động trong vườn thú hoặc công viên tự nhiên
        - Phát hiện động vật xâm nhập khu vực đô thị
        - Hỗ trợ quản lý trang trại và chăn nuôi
        """)
            
    # Add usage instructions
    with st.expander("📋 Hướng dẫn sử dụng", expanded=False):
        st.markdown("""
        ### Hướng dẫn sử dụng
        
        #### Cách sử dụng tính năng nhận dạng động vật:
        
        1. **Tải ảnh lên**
           - Nhấn nút "Browse files" để chọn ảnh từ thiết bị của bạn
           - Hỗ trợ các định dạng: JPG, JPEG, PNG, WEBP
           - Ảnh có thể chứa một hoặc nhiều động vật
        
        2. **Chọn thẻ phân loại**
           - Sử dụng menu chọn thẻ để lọc các loài động vật bạn muốn hiển thị
           - Có thể chọn nhiều loài cùng lúc
           - Chọn "Tất cả" để hiển thị tất cả các loài động vật phát hiện được
        
        3. **Xem kết quả nhận dạng**
           - Ảnh gốc sẽ hiển thị bên trái
           - Ảnh kết quả với các khung đánh dấu động vật được lọc sẽ hiển thị bên phải
           - Thông tin về loài động vật và độ tin cậy sẽ hiển thị bên dưới
        
        #### Lưu ý khi sử dụng:
        
        - **Ánh sáng**: Ảnh chụp cần có đủ ánh sáng để nhận diện tốt
        - **Góc nhìn**: Mô hình hoạt động tốt nhất khi động vật được nhìn thấy rõ ràng
        - **Hiệu suất**: Mô hình có thể nhận diện nhiều động vật cùng lúc trong một ảnh
        - **Giới hạn**: Hiệu quả nhất với động vật phổ biến và trong điều kiện ánh sáng tốt
        
        #### Mẹo cải thiện độ chính xác:
        
        - Chụp ảnh trong điều kiện ánh sáng tốt
        - Đảm bảo động vật chiếm phần đáng kể trong khung hình
        - Tránh các vật thể che khuất quá nhiều
        - Nếu có nhiều động vật, cố gắng để chúng không chồng lấn quá nhiều
        """)
    
    # Define specific animal classes we want to support for filtering
    animal_classes = ["dog", "cat", "bird", "horse", "cow", "elephant", "bear", "zebra", "giraffe", "tiger"]
    
    # Check if model exists
    model_exists = False
    animal_detector = None
    model_path = "models/animal_detection.pt"
    
    if os.path.exists(model_path):
        try:
            from utils.animal_detection import AnimalDetector
            animal_detector = AnimalDetector(model_path)
            model_exists = True
        except Exception as e:
            st.error(f"Lỗi khi tải model YOLOv8: {str(e)}")
            model_exists = False
    
    if not model_exists:
        st.error("Model YOLOv8 cho nhận dạng động vật chưa được cài đặt.")
        st.info("""
        **Để sử dụng chức năng này, bạn cần:**
        1. Tải model YOLOv8 được đào tạo cho nhận dạng động vật
        2. Đặt file model vào thư mục `models/` với tên `animal_detection.pt`
        3. Hoặc sử dụng lệnh sau để tải model từ Roboflow:
        ```
        from roboflow import Roboflow
        rf = Roboflow(api_key="API_KEY")
        project = rf.workspace("project_name").project("animal-detection")
        model = project.version(1).model
        model.save("models/animal_detection.pt")
        ```
        4. Bạn cũng có thể sử dụng script `scripts/download_animal_model.py` để tải model cơ bản:
        ```
        python scripts/download_animal_model.py
        ```
        5. Khởi động lại ứng dụng
        """)
        
    # Khởi tạo các biến session state nếu chưa có
    if 'selected_animal_tags' not in st.session_state:
        st.session_state.selected_animal_tags = animal_classes.copy()
        
    # Thêm widget chọn tag vào sidebar
    st.sidebar.markdown("### Bộ lọc loài động vật")
    
    # Tạo multiselect với tất cả các loài động vật
    selected_tags = st.sidebar.multiselect(
        "Chọn loài động vật cần hiển thị:",
        options=animal_classes,
        default=st.session_state.selected_animal_tags,
        help="Chọn các loài động vật bạn muốn hiển thị kết quả nhận dạng"
    )
    
    # Cập nhật session state khi có thay đổi
    st.session_state.selected_animal_tags = selected_tags
    
    # Thêm nút chọn/bỏ chọn tất cả
    col1, col2 = st.sidebar.columns(2)
    if col1.button("Chọn tất cả", key="select_all_animals"):
        st.session_state.selected_animal_tags = animal_classes.copy()
        st.rerun()
    if col2.button("Bỏ chọn tất cả", key="deselect_all_animals"):
        st.session_state.selected_animal_tags = []
        st.rerun()
    
    if model_exists:
        # Upload image
        uploaded_file = st.file_uploader("Chọn ảnh chứa động vật", type=["jpg", "jpeg", "png", "webp", "jfif", "tif", "tiff"])
        
        if uploaded_file is not None:
            # Safe image loading with error handling
            try:
                # Read image as bytes
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                
                # Check if file_bytes is not empty
                if len(file_bytes) == 0:
                    st.error("File tải lên rỗng hoặc bị lỗi.")
                else:
                    # Decode image
                    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    
                    # Check if image was decoded successfully
                    if img is None or img.size == 0:
                        st.error("Không thể đọc ảnh. Định dạng file có thể không được hỗ trợ.")
                    else:
                        # Create two columns for original and result images
                        col1, col2 = st.columns(2)
                        
                        # Display original image
                        with col1:
                            st.subheader("Ảnh gốc")
                            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)
                        
                        # Process image
                        with st.spinner("Đang nhận dạng động vật..."):
                            try:
                                # Phát hiện tất cả động vật
                                boxes, labels, scores = animal_detector.detect(img)
                                
                                # Lọc kết quả theo các tag đã chọn, chỉ giữ lại các nhãn thuộc animal_classes
                                valid_labels = []
                                valid_indices = []
                                
                                for i, label in enumerate(labels):
                                    # Kiểm tra xem nhãn có thuộc 10 loại động vật đã định nghĩa không
                                    if label in animal_classes:
                                        valid_labels.append(label)
                                        valid_indices.append(i)
                                
                                # Lấy chỉ các kết quả hợp lệ
                                if valid_indices:
                                    valid_boxes = boxes[valid_indices]
                                    valid_scores = [scores[i] for i in valid_indices]
                                else:
                                    valid_boxes = np.array([])
                                    valid_scores = []
                                
                                # Lọc tiếp dựa trên tag đã chọn
                                if selected_tags and valid_indices:
                                    filtered_indices = [i for i, label in enumerate(valid_labels) if label in selected_tags]
                                    filtered_boxes = valid_boxes[filtered_indices] if len(filtered_indices) > 0 else np.array([])
                                    filtered_labels = [valid_labels[i] for i in filtered_indices]
                                    filtered_scores = [valid_scores[i] for i in filtered_indices]
                                else:
                                    # Nếu không có tag nào được chọn, hiển thị ảnh không có bounding box
                                    filtered_boxes = np.array([])
                                    filtered_labels = []
                                    filtered_scores = []
                                
                                # Vẽ kết quả đã lọc lên ảnh
                                result_img = animal_detector.draw_results(img, filtered_boxes, filtered_labels, filtered_scores)
                                
                                # Display result
                                with col2:
                                    st.subheader("Kết quả")
                                    st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), use_container_width=True)
                                
                                # Hiển thị thông tin
                                if len(valid_labels) > 0:
                                    st.markdown("### Động vật đã phát hiện:")
                                    
                                    # Tạo bảng tóm tắt với số lượng mỗi loại
                                    animal_count = {}
                                    for label in valid_labels:
                                        if label in animal_count:
                                            animal_count[label] += 1
                                        else:
                                            animal_count[label] = 1
                                    
                                    # Hiển thị bảng tóm tắt
                                    st.markdown("#### Tóm tắt:")
                                    col_count = 3  # Số cột trong lưới
                                    cols = st.columns(col_count)
                                    
                                    # Hiển thị tổng quan các loài động vật
                                    for i, (label, count) in enumerate(animal_count.items()):
                                        with cols[i % col_count]:
                                            # Thêm emoji dựa vào loài động vật
                                            emoji = "🐾"  # Mặc định
                                            if label.lower() == "dog":
                                                emoji = "🐕"
                                            elif label.lower() == "cat":
                                                emoji = "🐈"
                                            elif label.lower() == "bird":
                                                emoji = "🐦"
                                            elif label.lower() == "horse":
                                                emoji = "🐎"
                                            elif label.lower() == "cow":
                                                emoji = "🐄"
                                            elif label.lower() == "elephant":
                                                emoji = "🐘"
                                            elif label.lower() == "bear":
                                                emoji = "🐻"
                                            elif label.lower() == "zebra":
                                                emoji = "🦓"
                                            elif label.lower() == "giraffe":
                                                emoji = "🦒"
                                            elif label.lower() == "tiger":
                                                emoji = "🐅"
                                            
                                            # Hiển thị với trạng thái lọc
                                            status = " (hiển thị)" if label in selected_tags else " (đã lọc)"
                                            st.markdown(f"**{emoji} {label}:** {count}{status}")
                                    
                                    # Hiển thị chi tiết các động vật đã lọc
                                    if filtered_boxes.size > 0:
                                        st.markdown("#### Chi tiết động vật được hiển thị:")
                                        # Create a nice looking grid for animal results
                                        col_count = 3  # Number of columns in the grid
                                        display_cols = st.columns(col_count)
                                        
                                        # Display each detected animal with details
                                        for i, (label, score) in enumerate(zip(filtered_labels, filtered_scores)):
                                            with display_cols[i % col_count]:
                                                st.markdown(f"**{i+1}. {label}**")
                                                st.markdown(f"Độ tin cậy: {score:.2f}")
                                                
                                                # Add emoji based on animal type
                                                emoji = "🐾"  # Default
                                                if label.lower() == "dog":
                                                    emoji = "🐕"
                                                elif label.lower() == "cat":
                                                    emoji = "🐈"
                                                elif label.lower() == "bird":
                                                    emoji = "🐦"
                                                elif label.lower() == "horse":
                                                    emoji = "🐎"
                                                elif label.lower() == "cow":
                                                    emoji = "🐄"
                                                elif label.lower() == "elephant":
                                                    emoji = "🐘"
                                                elif label.lower() == "bear":
                                                    emoji = "🐻"
                                                elif label.lower() == "zebra":
                                                    emoji = "🦓"
                                                elif label.lower() == "giraffe":
                                                    emoji = "🦒"
                                                elif label.lower() == "tiger":
                                                    emoji = "🐅"
                                                
                                                st.markdown(f"{emoji} {label}")
                                    else:
                                        if selected_tags:
                                            st.warning(f"Không phát hiện loài động vật nào trong các tag đã chọn: {', '.join(selected_tags)}")
                                        else:
                                            st.warning("Không có loài động vật nào được chọn để hiển thị. Vui lòng chọn ít nhất một loài.")
                                    
                                    # Display summary if there are matches
                                    if len(filtered_labels) > 0:
                                        st.markdown("### Tổng kết:")
                                        filtered_summary = {}
                                        for label in filtered_labels:
                                            if label in filtered_summary:
                                                filtered_summary[label] += 1
                                            else:
                                                filtered_summary[label] = 1
                                        
                                        summary_text = ", ".join([f"**{count} {label}**" for label, count in filtered_summary.items()])
                                        st.markdown(f"Đã phát hiện và hiển thị {summary_text} trong ảnh.")
                                else:
                                    st.warning("Không phát hiện động vật nào trong ảnh!")
                                    st.markdown("""
                                    **Gợi ý:**
                                    - Thử tải lên ảnh khác có động vật rõ ràng hơn
                                    - Đảm bảo động vật không bị che khuất quá nhiều
                                    - Chọn ảnh có điều kiện ánh sáng tốt hơn
                                    """)
                            except Exception as e:
                                st.error(f"Lỗi khi xử lý ảnh: {str(e)}")
                                st.info("Thử tải lên ảnh khác hoặc kiểm tra lại model và thư viện.")
                                
            except Exception as e:
                st.error(f"Lỗi khi xử lý file: {str(e)}")
                st.info("Vui lòng thử lại với file ảnh khác.")