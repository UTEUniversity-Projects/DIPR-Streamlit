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
    uploaded_file = st.file_uploader("Chọn ảnh để xử lý", type=["jpg", "jpeg", "png", "tif", "tiff"])
    
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