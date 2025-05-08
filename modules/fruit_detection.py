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
        
        2. **Xem kết quả nhận dạng**
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
    except Exception:
        has_fruit_detector = False
    
    if not has_fruit_detector:
        st.error("Model YOLOv8 chưa được cài đặt.")
        st.info("""
        **Để sử dụng chức năng này, bạn cần:**
        1. Huấn luyện model YOLOv8 theo hướng dẫn trong README.md
        2. Đặt file `fruit_detection.pt` vào thư mục `models/`
        3. Khởi động lại ứng dụng
        """)
    else:
        uploaded_file = st.file_uploader("Chọn ảnh chứa trái cây", type=["jpg", "jpeg", "png", "webp", "jfif", "tif", "tiff"])
        
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Ảnh gốc")
                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)
            
            with st.spinner("Đang nhận dạng trái cây..."):
                boxes, labels, scores = fruit_detector.detect(img)
                result_img = fruit_detector.draw_results(img, boxes, labels, scores)
                
                with col2:
                    st.subheader("Kết quả")
                    st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), use_container_width=True)
                
                if len(boxes) > 0:
                    st.markdown("### Trái cây phát hiện được:")
                    for i, (label, score) in enumerate(zip(labels, scores)):
                        st.write(f"**{i+1}.** {label} - Độ tin cậy: {score:.2f}")
                else:
                    st.warning("Không phát hiện trái cây nào trong ảnh!")