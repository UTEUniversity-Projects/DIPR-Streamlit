import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

@st.cache_resource
def load_image_processor():
    from utils.image_processing import ImageProcessor
    return ImageProcessor()

def show():
    # Section de introducción
    with st.expander("🔍 Giới thiệu về xử lý ảnh số", expanded=False):
        st.markdown("""
        ### Giới thiệu về xử lý ảnh số
        
        Tính năng xử lý ảnh số tích hợp các phương pháp xử lý từ 3 chương quan trọng trong lĩnh vực này:
        
        #### Chương 3: Các phép toán điểm ảnh
        Xử lý hình ảnh ở mức pixel, áp dụng các biến đổi toán học cho từng điểm ảnh riêng biệt:
        - **Negative**: Đảo âm bản ảnh bằng cách lấy giá trị 255 - giá trị pixel
        - **Logarit/Gamma**: Biến đổi phi tuyến để tăng cường chi tiết vùng tối/sáng
        - **Histogram**: Phân tích và cân bằng phân phối cường độ màu trong ảnh
        - **Làm nét cạnh**: Tăng cường chi tiết và viền bằng convolution
        
        #### Chương 4: Xử lý trong miền tần số
        Áp dụng biến đổi Fourier để xử lý ảnh trong miền tần số:
        - **Spectrum**: Hiển thị phổ tần số của ảnh
        - **Lọc thông cao/thấp**: Loại bỏ nhiễu và mẫu lặp lại không mong muốn
        - **Khử mờ chuyển động**: Phục hồi ảnh bị mờ do chuyển động
        
        #### Chương 9: Xử lý hình thái học
        Thao tác với hình dạng và cấu trúc của đối tượng trong ảnh:
        - **Erosion (Co)**: Thu nhỏ đối tượng, loại bỏ chi tiết nhỏ
        - **Dilation (Giãn)**: Mở rộng đối tượng, lấp đầy lỗ hổng
        - **Phát hiện đường biên**: Xác định và hiển thị đường viền của đối tượng
        
        Các phương pháp này có ứng dụng rộng rãi trong:
        - Xử lý ảnh y tế và phân tích hình ảnh khoa học
        - Cải thiện chất lượng ảnh và khôi phục ảnh
        - Trích xuất đặc trưng cho hệ thống thị giác máy tính
        - Xử lý tiền ảnh cho các thuật toán AI
        """)
            
    # Section de instrucciones
    with st.expander("📋 Hướng dẫn sử dụng", expanded=False):
        st.markdown("""
        ### Hướng dẫn sử dụng
        
        #### Cách thực hiện xử lý ảnh:
        
        1. **Chọn chương xử lý**
           - Chương 3: Xử lý điểm ảnh (Điểm Ảnh)
           - Chương 4: Xử lý tần số
           - Chương 9: Xử lý hình thái
        
        2. **Chọn phép toán**
           - Mỗi chương sẽ hiển thị danh sách các phép toán có thể áp dụng
           - Chọn một phép toán phù hợp với nhu cầu xử lý của bạn
        
        3. **Tải ảnh lên**
           - Nhấn "Browse files" để tải ảnh từ thiết bị
           - Hỗ trợ các định dạng: JPG, JPEG, PNG, TIF, TIFF
           - Ảnh sẽ được hiển thị bên trái màn hình
        
        4. **Xem kết quả và tải xuống**
           - Kết quả xử lý sẽ hiển thị bên phải
           - Nhấn "Tải xuống kết quả" để lưu ảnh đã xử lý
           - Chọn định dạng tải xuống: JPG, PNG hoặc TIFF
        """)
            
    image_processor = load_image_processor()
    available_functions = image_processor.get_available_functions()
    
    chapter = st.sidebar.selectbox("Chọn chương", ["3", "4", "9"])
    
    chapter_key = f"Chapter {chapter} - " + {
        "3": "Điểm Ảnh",
        "4": "Xử lý tần số",
        "9": "Xử lý hình thái"
    }[chapter]
    
    functions = available_functions[chapter_key]
    func_names = [f["name"] for f in functions]
    func_descriptions = {f["name"]: f["description"] for f in functions}
    
    selected_func = st.sidebar.selectbox(
        "Chọn phép toán",
        func_names,
        format_func=lambda x: f"{x} - {func_descriptions[x]}"
    )
    
    # Thêm hỗ trợ cho định dạng TIFF
    uploaded_file = st.file_uploader("Chọn ảnh để xử lý", type=["jpg", "jpeg", "png", "webp", "jfif", "tif", "tiff"])
    
    if uploaded_file is not None:
        # Kiểm tra định dạng file
        file_extension = uploaded_file.name.lower().split('.')[-1]
        
        if file_extension in ['tif', 'tiff']:
            # Đọc file TIFF bằng PIL và chuyển sang OpenCV
            img = Image.open(uploaded_file)
            # Chuyển đổi PIL image sang OpenCV format (PIL là RGB, OpenCV là BGR)
            img_np = np.array(img)
            
            # Kiểm tra số kênh màu
            if len(img_np.shape) == 3 and img_np.shape[2] == 3:
                # Ảnh RGB, chuyển sang BGR cho OpenCV
                img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            elif len(img_np.shape) == 3 and img_np.shape[2] == 4:
                # Ảnh RGBA, chuyển sang BGR cho OpenCV
                img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
            else:
                # Ảnh grayscale, giữ nguyên
                img_cv = img_np
        else:
            # Đọc file không phải TIFF bằng OpenCV
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img_cv = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Hiển thị ảnh gốc
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Ảnh gốc")
            # Chuyển đổi từ BGR sang RGB trước khi hiển thị với Streamlit
            if len(img_cv.shape) == 3 and img_cv.shape[2] == 3:
                display_img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                st.image(display_img, use_container_width=True)
            else:
                # Ảnh grayscale, không cần chuyển đổi
                st.image(img_cv, use_container_width=True)
        
        with st.spinner(f"Đang áp dụng {selected_func}..."):
            try:
                if selected_func in ["NegativeColor", "HistEqualColor"]:
                    processed_img = image_processor.process(img_cv, chapter, selected_func)
                else:
                    if len(img_cv.shape) == 3:
                        gray_img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
                    else:
                        gray_img = img_cv
                    processed_img = image_processor.process(gray_img, chapter, selected_func)
                
                with col2:
                    st.subheader("Kết quả")
                    # Hiển thị ảnh kết quả tùy thuộc vào loại ảnh
                    if len(processed_img.shape) == 2:  # Ảnh xám
                        st.image(processed_img, use_container_width=True)
                    else:  # Ảnh màu (BGR)
                        # Chuyển đổi từ BGR sang RGB để hiển thị với Streamlit
                        display_processed = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
                        st.image(display_processed, use_container_width=True)
                
                if st.button("Tải xuống kết quả"):
                    # Tùy chọn định dạng khi tải xuống
                    download_format = st.select_slider(
                        "Chọn định dạng:",
                        options=['JPG', 'PNG', 'TIFF'],
                        value='JPG'
                    )
                    
                    # Tải xuống với định dạng được chọn
                    if download_format == 'TIFF':
                        # Chuyển đổi và lưu là TIFF
                        if len(processed_img.shape) == 2:
                            # Ảnh grayscale
                            pil_img = Image.fromarray(processed_img)
                        else:
                            # Ảnh BGR, chuyển sang RGB cho PIL
                            pil_img = Image.fromarray(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB))
                        
                        buf = io.BytesIO()
                        pil_img.save(buf, format='TIFF')
                        tiff_data = buf.getvalue()
                        
                        st.download_button(
                            label=f"Download {download_format}",
                            data=tiff_data,
                            file_name=f"processed_{selected_func}.tiff",
                            mime="image/tiff"
                        )
                    else:
                        # Lưu JPG hoặc PNG
                        ext = '.jpg' if download_format == 'JPG' else '.png'
                        
                        # OpenCV lưu file trực tiếp ở định dạng BGR
                        _, buffer = cv2.imencode(ext, processed_img)
                        
                        st.download_button(
                            label=f"Download {download_format}",
                            data=buffer.tobytes(),
                            file_name=f"processed_{selected_func}{ext}",
                            mime=f"image/{download_format.lower()}"
                        )
                    
            except Exception as e:
                st.error(f"Lỗi xử lý ảnh: {str(e)}")