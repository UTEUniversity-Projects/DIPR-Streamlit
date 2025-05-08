import streamlit as st
import cv2
import numpy as np

def show():
    # Thêm phần giới thiệu
    with st.expander("🔍 Giới thiệu về nhận dạng trái cây", expanded=False):
        st.markdown("""
        ### Giới thiệu về nhận dạng trái cây
        
        Tính năng nhận dạng trái cây sử dụng mô hình YOLOv8 (You Only Look Once), một trong những phương pháp phát hiện đối tượng hiện đại và hiệu quả nhất hiện nay.
        
        #### Kiến trúc YOLOv8
        
        YOLOv8 là phiên bản mới nhất của YOLO với nhiều cải tiến so với các phiên bản trước:
        
        - **Backbone**: CSPDarknet với cải tiến giúp trích xuất đặc trưng tốt hơn
        - **Neck**: PANet (Path Aggregation Network) cải tiến giúp truyền thông tin giữa các tầng đặc trưng
        - **Head**: Đầu ra bao gồm các dự đoán về vị trí (bounding box) và lớp đối tượng
        - **Loss Function**: CIoU (Complete IoU) cho dự đoán bounding box tối ưu hơn
        
        #### Huấn luyện cho nhận dạng trái cây
        
        Mô hình được huấn luyện trên dataset trái cây với 5 lớp:
        1. Apple (Táo)
        2. Banana (Chuối)
        3. Kiwi
        4. Orange (Cam)
        5. Pear (Lê)
        
        Quá trình huấn luyện được thực hiện trên Google Colab với GPU để tăng tốc, sử dụng các kỹ thuật:
        - Data augmentation (xoay, lật, thay đổi màu sắc)
        - Transfer learning từ mô hình pre-trained
        - Early stopping và model checkpointing
        
        #### Ứng dụng trong thực tế
        
        Nhận dạng trái cây có nhiều ứng dụng thực tế:
        - Hệ thống thanh toán tự động tại siêu thị
        - Phân loại chất lượng trái cây trong nông nghiệp
        - Hỗ trợ robot thu hoạch trong nông nghiệp thông minh
        - Phân tích dinh dưỡng tự động dựa trên nhận dạng thực phẩm
        """)
            
    # Thêm phần hướng dẫn
    with st.expander("📋 Hướng dẫn sử dụng", expanded=False):
        st.markdown("""
        ### Hướng dẫn sử dụng
        
        #### Cách sử dụng tính năng nhận dạng trái cây:
        
        1. **Tải ảnh lên**
           - Nhấn nút "Browse files" để chọn ảnh từ thiết bị của bạn
           - Hỗ trợ các định dạng: JPG, JPEG, PNG, WEBP
           - Kích thước ảnh tối ưu: 640x640 pixels
        
        2. **Lựa chọn thẻ**
           - Sử dụng menu chọn thẻ để lọc các loại trái cây bạn muốn hiển thị
           - Có thể chọn nhiều loại cùng lúc
           - Chọn "Tất cả" để hiển thị tất cả các loại trái cây phát hiện được
        
        3. **Xem kết quả nhận dạng**
           - Ảnh gốc sẽ hiển thị bên trái
           - Ảnh kết quả với các bounding box sẽ hiển thị bên phải
           - Thông tin về loại trái cây và độ tin cậy sẽ hiển thị bên dưới
        
        #### Lưu ý khi sử dụng:
        
        - **Ánh sáng**: Ảnh chụp cần có đủ ánh sáng để nhận diện tốt
        - **Góc nhìn**: Nên chụp trái cây ở góc nhìn rõ ràng, không bị che khuất
        - **Nhiều đối tượng**: Mô hình có thể nhận dạng nhiều trái cây cùng lúc trong một ảnh
        - **Độ tin cậy**: Kết quả với độ tin cậy (confidence) cao hơn đáng tin cậy hơn
        
        #### Mẹo cải thiện độ chính xác:
        
        - Chụp ảnh rõ nét, không bị mờ
        - Đặt trái cây trên nền đơn giản, tương phản với màu của trái cây
        - Tránh chụp trong điều kiện ánh sáng quá tối hoặc quá sáng
        - Mỗi ảnh nên chứa tối đa 10 trái cây để có kết quả tốt nhất
        """)
            
    try:
        from utils.fruit_detection import FruitDetector
        fruit_detector = FruitDetector("models/fruit_detection.pt")
        has_fruit_detector = True
        
        # Lấy danh sách các loại trái cây từ detector
        fruit_classes = list(fruit_detector.class_names.values())
    except Exception:
        has_fruit_detector = False
        fruit_classes = ["Apple", "Banana", "Kiwi", "Orange", "Pear"]
    
    if not has_fruit_detector:
        st.error("Model YOLOv8 chưa được cài đặt.")
        st.info("""
        **Để sử dụng chức năng này, bạn cần:**
        1. Huấn luyện model YOLOv8 theo hướng dẫn trong README.md
        2. Đặt file `fruit_detection.pt` vào thư mục `models/`
        3. Khởi động lại ứng dụng
        """)
    else:
        # Thêm widget chọn tag
        st.sidebar.markdown("### Bộ lọc loại trái cây")
        
        # Khởi tạo các biến session state nếu chưa có
        if 'selected_fruit_tags' not in st.session_state:
            st.session_state.selected_fruit_tags = fruit_classes.copy()
        
        # Tạo multiselect với tất cả các loại trái cây
        selected_tags = st.sidebar.multiselect(
            "Chọn loại trái cây cần hiển thị:",
            options=fruit_classes,
            default=st.session_state.selected_fruit_tags,
            help="Chọn các loại trái cây bạn muốn hiển thị kết quả nhận dạng"
        )
        
        # Cập nhật session state khi có thay đổi
        st.session_state.selected_fruit_tags = selected_tags
        
        # Thêm nút chọn/bỏ chọn tất cả
        col1, col2 = st.sidebar.columns(2)
        if col1.button("Chọn tất cả"):
            st.session_state.selected_fruit_tags = fruit_classes.copy()
            st.rerun()
        if col2.button("Bỏ chọn tất cả"):
            st.session_state.selected_fruit_tags = []
            st.rerun()
        
        uploaded_file = st.file_uploader("Chọn ảnh chứa trái cây", type=["jpg", "jpeg", "png", "webp", "jfif", "tif", "tiff"])
        
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Ảnh gốc")
                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)
            
            with st.spinner("Đang nhận dạng trái cây..."):
                # Phát hiện tất cả trái cây
                boxes, labels, scores = fruit_detector.detect(img)
                
                # Lọc kết quả theo các tag đã chọn
                if selected_tags:
                    filtered_indices = [i for i, label in enumerate(labels) if label in selected_tags]
                    filtered_boxes = boxes[filtered_indices] if len(filtered_indices) > 0 else np.array([])
                    filtered_labels = [labels[i] for i in filtered_indices]
                    filtered_scores = [scores[i] for i in filtered_indices]
                else:
                    # Nếu không có tag nào được chọn, hiển thị ảnh không có bounding box
                    filtered_boxes = np.array([])
                    filtered_labels = []
                    filtered_scores = []
                
                # Vẽ kết quả đã lọc lên ảnh
                result_img = fruit_detector.draw_results(img, filtered_boxes, filtered_labels, filtered_scores)
                
                with col2:
                    st.subheader("Kết quả")
                    st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), use_container_width=True)
                
                # Hiển thị thông tin về trái cây được phát hiện
                if len(boxes) > 0:
                    st.markdown("### Trái cây phát hiện được:")
                    
                    # Tạo bảng tóm tắt với số lượng mỗi loại
                    summary = {}
                    for label in labels:
                        if label in summary:
                            summary[label] += 1
                        else:
                            summary[label] = 1
                    
                    # Hiển thị bảng tóm tắt
                    st.markdown("#### Tóm tắt:")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Loại trái cây**")
                        for label in summary:
                            st.markdown(f"- {label}")
                    with col2:
                        st.markdown("**Số lượng**")
                        for label in summary:
                            count = summary[label]
                            if label in selected_tags:
                                st.markdown(f"- {count} (hiển thị)")
                            else:
                                st.markdown(f"- {count} (đã lọc)")
                    
                    # Hiển thị chi tiết các trái cây được lọc
                    if filtered_boxes.size > 0:
                        st.markdown("#### Chi tiết trái cây được hiển thị:")
                        for i, (label, score) in enumerate(zip(filtered_labels, filtered_scores)):
                            st.write(f"**{i+1}.** {label} - Độ tin cậy: {score:.2f}")
                    else:
                        if selected_tags:
                            st.warning(f"Không phát hiện loại trái cây nào trong các tag đã chọn: {', '.join(selected_tags)}")
                        else:
                            st.warning("Không có loại trái cây nào được chọn để hiển thị. Vui lòng chọn ít nhất một loại trái cây.")
                else:
                    st.warning("Không phát hiện trái cây nào trong ảnh!")