import streamlit as st
import cv2
import numpy as np

def show():
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
        uploaded_file = st.file_uploader("Chọn ảnh chứa trái cây", type=["jpg", "jpeg", "png","webp"])
        
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