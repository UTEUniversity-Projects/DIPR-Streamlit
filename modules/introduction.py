import streamlit as st

def show():
    """Display the introduction page"""
    st.markdown("## 👋 Giới thiệu về ứng dụng Xử lý Ảnh Số")
    
    # Main introduction from README
    st.markdown("""
    Ứng dụng web Xử lý Ảnh Số sử dụng Streamlit, OpenCV và deep learning cho các bài toán nhận dạng khuôn mặt, 
    nhận dạng trái cây, nhận dạng động vật, và các phép toán xử lý ảnh.
    
    Ứng dụng này được phát triển như một dự án cho môn học Xử lý Ảnh Số, tích hợp nhiều kỹ thuật tiên tiến 
    trong lĩnh vực thị giác máy tính và xử lý ảnh.
    """)
    
    # Overview of key features
    st.markdown("### 🌟 Tính năng chính")
    
    # Face Recognition
    with st.expander("**1. Nhận dạng khuôn mặt** - Sử dụng mô hình ONNX hiện đại", expanded=True):
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image("https://opencv.org/wp-content/uploads/2021/01/OpenCV-logo.png", width=150)
        with col2:
            st.markdown("""
            - Sử dụng 2 mô hình ONNX: YuNet (phát hiện) và SFace (nhận dạng)
            - Hỗ trợ chụp ảnh từ webcam để xây dựng database
            - Mỗi người chụp 100 ảnh mẫu
            - Nhận dạng tối đa 5 người trong một khung hình
            """)
    
    # Fruit Detection
    with st.expander("**2. Nhận dạng trái cây** - Sử dụng YOLOv8"):
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image("https://media.springernature.com/lw685/springer-static/image/art%3A10.1007%2Fs11119-023-10009-9/MediaObjects/11119_2023_10009_Fig12_HTML.png", width=350)
        with col2:
            st.markdown("""
            - Sử dụng YOLOv8 để nhận dạng 5 loại trái cây
            - Hướng dẫn huấn luyện trên Google Colab
            - Dataset trái cây có thể mở rộng
            - Hiển thị bounding box và độ tin cậy
            """)
    
    # Animal Detection
    with st.expander("**3. Nhận dạng động vật** - Phát hiện và phân loại"):
        st.markdown("""
        - Sử dụng YOLOv8 để nhận dạng nhiều loại động vật phổ biến
        - Hiển thị độ tin cậy và thông tin chi tiết về từng động vật
        - Hỗ trợ nhận dạng nhiều động vật trong một khung hình
        - Hiển thị emoji và thông tin tổng kết về các động vật đã phát hiện
        """)
    
    # Digital Image Processing
    with st.expander("**4. Xử lý ảnh số** - Các phép toán xử lý ảnh"):
        st.markdown("""
        - **Chương 3**: Các phép toán điểm ảnh (Negative, Logarit, Gamma, Histogram, v.v.)
        - **Chương 4**: Xử lý tần số (Spectrum, RemoveMoire, DeMotion, v.v.)
        - **Chương 9**: Xử lý hình thái (Erosion, Dilation, Boundary, Contour)
        - Hỗ trợ nhiều định dạng ảnh đầu vào và đầu ra
        """)
    
    # 3D KITTI
    with st.expander("**5. Nhận dạng 3D KITTI** - Phân tích dữ liệu LiDAR"):
        st.markdown("""
        - Sử dụng kiến trúc PointPillars để nhận dạng đối tượng từ dữ liệu LiDAR
        - Hiển thị kết quả trực quan bằng đồ họa 3D tương tác
        - Hỗ trợ nhận dạng xe hơi, người đi bộ và xe đạp
        - Kết hợp dữ liệu LiDAR và camera để trực quan hóa
        """)
    
    # System requirements
    st.markdown("### 💻 Yêu cầu hệ thống")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Phần cứng tối thiểu:**
        - CPU: 2 nhân
        - RAM: 4GB
        - Ổ cứng: 2GB dung lượng trống
        - Webcam (cho chức năng chụp ảnh)
        """)
    
    with col2:
        st.markdown("""
        **Phần mềm:**
        - Python 3.7+
        - Các thư viện trong requirements.txt
        - GPU (tùy chọn, cho huấn luyện YOLOv8)
        """)
    
    # About the author
    st.markdown("### 👨‍💻 Thông tin tác giả")
    st.info("**Sinh viên:** Lê Hồng Phúc - **MSSV:** 22110399")
    