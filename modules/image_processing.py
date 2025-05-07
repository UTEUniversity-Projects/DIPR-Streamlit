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
    # Sección de introducción
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
            
    # Sección de instrucciones
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
        
        #### Gợi ý cho từng loại xử lý:
        
        **Xử lý điểm ảnh (Chương 3):**
        - **Negative**: Tạo hiệu ứng âm bản
        - **Logarit/Gamma**: Điều chỉnh độ sáng và tăng cường chi tiết vùng tối
        - **HistEqual**: Cải thiện độ tương phản
        - **Sharpening**: Làm sắc nét ảnh mờ
        
        **Xử lý tần số (Chương 4):**
        - **Spectrum**: Phân tích phổ tần số của ảnh
        - **RemoveMoire**: Loại bỏ hiệu ứng Moire trong ảnh scan
        - **RemoveInterference**: Loại bỏ nhiễu giao thoa
        - **DeMotion**: Khử mờ chuyển động
        
        **Xử lý hình thái (Chương 9):**
        - **Erosion**: Loại bỏ chi tiết nhỏ, làm mỏng đối tượng
        - **Dilation**: Làm dày đối tượng, lấp đầy lỗ hổng nhỏ
        - **Boundary**: Phát hiện biên của đối tượng
        - **Contour**: Tạo đường viền cho các đối tượng trong ảnh
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
    uploaded_file = st.file_uploader("Chọn ảnh để xử lý", type=["jpg", "jpeg", "png", "webp", "avif", "tif", "tiff"])
    
    if uploaded_file is not None:
        # Kiểm tra định dạng file
        file_extension = uploaded_file.name.lower().split('.')[-1]
        
        if file_extension in ['tif', 'tiff']:
            # Đọc file TIFF bằng PIL và chuyển sang OpenCV
            img = Image.open(uploaded_file)
            # Chuyển đổi PIL image sang OpenCV format
            if img.mode == 'RGB':
                img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            else:
                # Nếu là grayscale hoặc CMYK, chuyển sang RGB trước
                img_cv = cv2.cvtColor(np.array(img.convert('RGB')), cv2.COLOR_RGB2BGR)
        else:
            # Đọc file không phải TIFF bằng OpenCV
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img_cv = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Hiển thị ảnh gốc
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Ảnh gốc")
            st.image(img_cv, channels="BGR", use_container_width=True)
        
        with st.spinner(f"Đang áp dụng {selected_func}..."):
            try:
                if selected_func in ["NegativeColor", "HistEqualColor"]:
                    processed_img = image_processor.process(img_cv, chapter, selected_func)
                else:
                    gray_img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
                    processed_img = image_processor.process(gray_img, chapter, selected_func)
                
                with col2:
                    st.subheader("Kết quả")
                    # Sửa lại: không dùng 'cmap' cho ảnh grayscale
                    if len(processed_img.shape) == 2:  # Ảnh xám
                        # Chuyển ảnh grayscale về RGB để hiển thị
                        processed_img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2RGB)
                        st.image(processed_img_rgb, use_container_width=True)
                    else:  # Ảnh màu
                        st.image(processed_img, channels="BGR", use_container_width=True)
                
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
                            pil_img = Image.fromarray(processed_img)
                        else:
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
                        _, buffer = cv2.imencode(ext, processed_img)
                        st.download_button(
                            label=f"Download {download_format}",
                            data=buffer.tobytes(),
                            file_name=f"processed_{selected_func}{ext}",
                            mime=f"image/{download_format.lower()}"
                        )
                    
            except Exception as e:
                st.error(f"Lỗi xử lý ảnh: {str(e)}")