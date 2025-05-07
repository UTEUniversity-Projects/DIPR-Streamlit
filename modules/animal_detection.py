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
        
        2. **Xem kết quả nhận dạng**
           - Ảnh gốc sẽ hiển thị bên trái
           - Ảnh kết quả với các khung đánh dấu động vật sẽ hiển thị bên phải
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
        
        # Cung cấp phần download model mẫu
        st.markdown("### Tải xuống model mẫu")
        st.markdown("""
        Bạn có thể tải model mẫu từ các nguồn sau:
        - [Roboflow Universe - Animal Detection](https://universe.roboflow.com/animal-detection-using-yolov8/animal-detection-using-yolov8)
        - [Hugging Face - YOLOv8 Animal Detection](https://huggingface.co/models?search=yolov8+animal)
        
        Sau khi tải xuống, hãy đổi tên và đặt file vào thư mục `models/animal_detection.pt`
        """)
    else:
        # Upload image
        uploaded_file = st.file_uploader("Chọn ảnh chứa động vật", type=["jpg", "jpeg", "png", "webp", "avif"])
        
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
                                boxes, labels, scores = animal_detector.detect(img)
                                result_img = animal_detector.draw_results(img, boxes, labels, scores)
                                
                                # Display result
                                with col2:
                                    st.subheader("Kết quả")
                                    st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), use_container_width=True)
                                
                                # Display information
                                if len(boxes) > 0:
                                    st.markdown("### Động vật đã phát hiện:")
                                    
                                    # Create a nice looking grid for animal results
                                    animal_count = {}
                                    for label in labels:
                                        if label in animal_count:
                                            animal_count[label] += 1
                                        else:
                                            animal_count[label] = 1
                                    
                                    # Create a grid of animals
                                    col_count = 3  # Number of columns in the grid
                                    cols = st.columns(col_count)
                                    
                                    # Display each detected animal with details
                                    for i, (label, score) in enumerate(zip(labels, scores)):
                                        with cols[i % col_count]:
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
                                    
                                    # Display summary
                                    st.markdown("### Tổng kết:")
                                    summary_text = ", ".join([f"**{count} {label}**" for label, count in animal_count.items()])
                                    st.markdown(f"Đã phát hiện {summary_text} trong ảnh.")
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