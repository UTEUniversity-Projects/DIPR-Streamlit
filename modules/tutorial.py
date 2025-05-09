import streamlit as st

def show():
    """Display the tutorial page with detailed instructions for each module"""
    st.markdown("## 📚 Hướng dẫn sử dụng ứng dụng")
    
    st.markdown("""
    Chào mừng bạn đến với hướng dẫn sử dụng ứng dụng Xử lý Ảnh Số. Trang này cung cấp 
    hướng dẫn chi tiết cho mỗi tính năng của ứng dụng. Hãy chọn một tính năng từ menu bên dưới 
    để xem hướng dẫn cụ thể.
    """)
    
    # Select feature to view tutorial
    selected_feature = st.selectbox(
        "Chọn tính năng:",
        [
            "Cài đặt và khởi động",
            "Nhận dạng khuôn mặt",
            "Nhận dạng trái cây", 
            "Nhận dạng động vật",
            "Xử lý ảnh số",
            "Đăng ký khuôn mặt mới",
            "Nhận dạng 3D KITTI"
        ]
    )
    
    st.markdown("---")
    
    # Installation and startup
    if selected_feature == "Cài đặt và khởi động":
        st.markdown("### 🔧 Cài đặt và khởi động ứng dụng")
        
        # Installation steps
        with st.expander("Yêu cầu hệ thống", expanded=True):
            st.markdown("""
            **Yêu cầu tối thiểu:**
            - Python 3.7 trở lên
            - Webcam (cho chức năng chụp ảnh)
            - GPU (tùy chọn, cho huấn luyện YOLOv8)
            """)
        
        with st.expander("Cài đặt thư viện"):
            st.code("""
            # Tạo môi trường ảo (khuyến nghị)
            python -m venv venv
            
            # Kích hoạt môi trường ảo
            # Windows
            venv\\Scripts\\activate
            # Linux/MacOS
            source venv/bin/activate
            
            # Cài đặt thư viện
            pip install -r requirements.txt
            """, language="bash")
            
            st.markdown("**Nội dung file requirements.txt:**")
            st.code("""numpy>=1.21.0
opencv-python>=4.5.5
streamlit>=1.18.0
ultralytics>=8.0.0
matplotlib>=3.5.0
scikit-learn>=1.0.0
pillow>=9.0.0
scipy>=1.7.0
onnxruntime>=1.10.0
pyyaml>=6.0.0
tqdm>=4.62.0
roboflow>=1.1.9
# KITTI 3D dependencies
torch>=2.0.0
torchvision>=0.15.0
open3d>=0.17.0
pyquaternion>=0.9.0
mayavi>=4.8.1
plotly>=5.14.0
transformations>=2023.2.4
argparse>=1.4.0""", language="text")
        
        with st.expander("Tải các mô hình"):
            st.markdown("""
            ### Bước 1: Tải mô hình face detection và recognition:
            - [face_detection_yunet_2023mar.onnx](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)
            - [face_recognition_sface_2021dec.onnx](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface)
            
            ### Bước 2: Đặt chúng vào thư mục `models/`
            
            ### Bước 3: Huấn luyện mô hình YOLOv8 hoặc sử dụng mô hình có sẵn
            - Tham khảo hướng dẫn huấn luyện trong phần "Nhận dạng trái cây" và "Nhận dạng động vật"
            """)
            
        with st.expander("Khởi động ứng dụng"):
            st.code("""
            # Khởi động ứng dụng
            streamlit run app.py
            """, language="bash")
            
            st.markdown("""
            Sau khi chạy lệnh trên, ứng dụng sẽ tự động mở trong trình duyệt web của bạn tại địa chỉ http://localhost:8501
            
            Nếu cần chỉ định cổng khác:
            ```bash
            streamlit run app.py --server.port 8000
            ```
            """)
            
    # Face Recognition
    elif selected_feature == "Nhận dạng khuôn mặt":
        st.markdown("### 🖼️ Hướng dẫn sử dụng tính năng Nhận dạng khuôn mặt")
        
        st.markdown("""
        Tính năng nhận dạng khuôn mặt cho phép bạn phát hiện và nhận dạng khuôn mặt trong ảnh tĩnh hoặc video.
        Hệ thống sử dụng kết hợp hai mô hình ONNX hiện đại: YuNet cho phát hiện khuôn mặt và SFace cho nhận dạng.
        """)
        
        with st.expander("Chuẩn bị"):
            st.markdown("""
            Trước khi sử dụng tính năng này, bạn cần:
            
            1. Đảm bảo đã cài đặt các mô hình:
               - `models/face_detection_yunet_2023mar.onnx`
               - `models/face_recognition_sface_2021dec.onnx`
               
            2. Đã đăng ký ít nhất một khuôn mặt trong hệ thống (xem phần "Đăng ký khuôn mặt mới")
               - Mỗi người cần có khoảng 100 ảnh mẫu để đạt độ chính xác cao
            """)
            
        with st.expander("Các chế độ nhận dạng"):
            st.markdown("""
            Tính năng này cung cấp 3 chế độ nhận dạng:
            
            1. **Ảnh tĩnh**:
               - Tải lên ảnh từ máy tính
               - Chụp ảnh từ webcam
               
            2. **Video tải lên**:
               - Tải lên file video (.mp4, .mov, .avi, v.v.)
               - Điều chỉnh tốc độ xử lý và ngưỡng nhận dạng
               - Hiển thị kết quả nhận dạng trực tiếp trên video
               - Tải xuống video đã xử lý
               
            3. **Video trực tiếp**:
               - Sử dụng webcam để nhận dạng trong thời gian thực
               - Điều chỉnh độ phân giải và tốc độ xử lý
            """)
            
        with st.expander("Hướng dẫn sử dụng chi tiết"):
            st.markdown("""
            #### 1. Chế độ ảnh tĩnh
            
            1. Chọn "📸 Ảnh tĩnh" ở phần radio button
            2. Chọn phương thức đầu vào:
               - **Upload ảnh**: Nhấn "Browse files" để tải lên ảnh từ máy tính
               - **Chụp từ webcam**: Nhấn "Take photo" để chụp ảnh từ webcam
            3. Sau khi có ảnh, hệ thống sẽ tự động phát hiện và nhận dạng khuôn mặt
            4. Kết quả hiển thị bên phải, kèm thông tin chi tiết bên dưới
            
            #### 2. Chế độ video tải lên
            
            1. Chọn "🎬 Video tải lên" ở phần radio button
            2. Nhấn "Browse files" để tải lên file video
            3. Điều chỉnh các thông số:
               - **Ngưỡng nhận diện**: Điều chỉnh độ nhạy (0.4-0.95)
               - **Tốc độ xử lý**: Điều chỉnh tốc độ (1-10)
            4. Nhấn "▶️ Bắt đầu" để bắt đầu xử lý video
            5. Có thể tạm dừng, tiếp tục hoặc dừng hoàn toàn quá trình
            6. Sau khi hoàn thành, có thể tải xuống video đã xử lý
            
            #### 3. Chế độ video trực tiếp
            
            1. Chọn "🎥 Video trực tiếp" ở phần radio button
            2. Điều chỉnh các thông số:
               - **Độ phân giải**: Chọn độ phân giải phù hợp
               - **Tốc độ xử lý**: Điều chỉnh số frame bỏ qua giữa các lần xử lý
            3. Nhấn "Bắt đầu" để khởi động webcam và bắt đầu nhận dạng
            4. Nhấn "Dừng" để kết thúc quá trình
            """)
            
        with st.expander("Mẹo để cải thiện kết quả"):
            st.markdown("""
            - **Ánh sáng**: Đảm bảo khuôn mặt được chiếu sáng đầy đủ và đều
            - **Góc nhìn**: Để khuôn mặt hướng thẳng vào camera nếu có thể
            - **Khoảng cách**: Giữ khoảng cách phù hợp, không quá gần hoặc quá xa
            - **Tốc độ xử lý**: Giảm tốc độ xử lý nếu hiệu suất máy tính không tốt
            - **Ngưỡng nhận diện**: Tăng ngưỡng nếu có quá nhiều nhận diện sai, giảm nếu không nhận diện được
            """)
        
    # Fruit Detection
    elif selected_feature == "Nhận dạng trái cây":
        st.markdown("### 🍎 Hướng dẫn sử dụng tính năng Nhận dạng trái cây")
        
        st.markdown("""
        Tính năng nhận dạng trái cây sử dụng mô hình YOLOv8 để phát hiện và phân loại 5 loại trái cây: táo, chuối, cam, kiwi, và lê.
        """)
        
        with st.expander("Chuẩn bị mô hình"):
            st.markdown("""
            Trước khi sử dụng tính năng này, bạn cần:
            
            1. Đảm bảo đã huấn luyện hoặc tải về mô hình YOLOv8 cho nhận dạng trái cây
            2. Đặt file mô hình `fruit_detection.pt` vào thư mục `models/`
            
            Nếu chưa có mô hình, bạn có thể:
            - Tải về mô hình đã huấn luyện sẵn
            - Huấn luyện mô hình mới sử dụng Google Colab (xem phần dưới đây)
            """)
            
        with st.expander("Huấn luyện mô hình trên Google Colab"):
            st.markdown("""
            ### Hướng dẫn huấn luyện mô hình YOLOv8 trên Google Colab
            
            1. Mở Google Colab tại [colab.research.google.com](https://colab.research.google.com)
            2. Tạo notebook mới
            3. Copy nội dung file `scripts/train_fruit_detection.py` vào Colab notebook
            4. Đảm bảo chọn runtime GPU: Runtime > Change runtime type > Hardware accelerator > GPU
            5. Chạy notebook để bắt đầu quá trình huấn luyện
            6. Sau khi huấn luyện xong, tải file `best.pt` về máy tính
            7. Đổi tên thành `fruit_detection.pt` và đặt vào thư mục `models/`
            """)
            
        with st.expander("Hướng dẫn sử dụng"):
            st.markdown("""
            #### Cách sử dụng tính năng nhận dạng trái cây:
            
            1. Chọn tính năng "🍎 Nhận dạng trái cây" từ menu bên trái
            2. Sử dụng bộ lọc loại trái cây ở sidebar (có thể chọn hiển thị tất cả hoặc chỉ một số loại)
            3. Nhấn "Browse files" để tải lên ảnh chứa trái cây
            4. Hệ thống sẽ tự động phát hiện và phân loại trái cây trong ảnh
            5. Kết quả hiển thị bên phải với các bounding box và tên loại trái cây
            6. Thông tin chi tiết về các trái cây được phát hiện hiển thị bên dưới
            """)
            
        with st.expander("Mẹo chụp ảnh để nhận dạng tốt hơn"):
            st.markdown("""
            - **Ánh sáng**: Chụp ảnh trong điều kiện ánh sáng tốt
            - **Nền**: Sử dụng nền đơn giản, tương phản với màu trái cây
            - **Góc nhìn**: Chụp trái cây ở góc nhìn rõ ràng, không bị che khuất
            - **Khoảng cách**: Không chụp quá xa hoặc quá gần
            - **Số lượng**: Mỗi ảnh nên chứa tối đa 10 trái cây để có kết quả tốt nhất
            - **Độ phân giải**: Ảnh chụp nên có độ phân giải tốt, không bị mờ
            """)
            
    # Animal Detection - CẬP NHẬT
    elif selected_feature == "Nhận dạng động vật":
        st.markdown("### 🐾 Hướng dẫn sử dụng tính năng Nhận dạng động vật")
        
        st.markdown("""
        Tính năng nhận dạng động vật sử dụng mô hình YOLOv8 để phát hiện và phân loại nhiều loài động vật phổ biến 
        như chó, mèo, ngựa, voi, v.v. Mô hình được huấn luyện trên Google Colab để đạt hiệu suất tốt nhất.
        """)
        
        with st.expander("Chuẩn bị mô hình"):
            st.markdown("""
            Trước khi sử dụng tính năng này, bạn cần:
            
            1. Đảm bảo đã huấn luyện hoặc tải về mô hình YOLOv8 cho nhận dạng động vật
            2. Đặt file mô hình `animal_detection.pt` vào thư mục `models/`
            
            Nếu chưa có mô hình, bạn có thể:
            - Tải về mô hình đã huấn luyện sẵn
            - Huấn luyện mô hình mới sử dụng Google Colab (xem phần dưới đây)
            """)
            
        with st.expander("Huấn luyện mô hình trên Google Colab"):
            st.markdown("""
            ### Hướng dẫn huấn luyện mô hình YOLOv8 trên Google Colab
            
            1. Mở Google Colab tại [colab.research.google.com](https://colab.research.google.com)
            2. Tạo notebook mới
            3. Copy nội dung file `scripts/train_animal_detection.py` vào Colab notebook
            4. Đảm bảo chọn runtime GPU: Runtime > Change runtime type > Hardware accelerator > GPU
            5. Chạy notebook để bắt đầu quá trình huấn luyện
              - Script sẽ tự động tải dataset động vật từ Roboflow
              - Quá trình huấn luyện sẽ diễn ra với 100 epochs
              - Kết quả huấn luyện sẽ được lưu vào Google Drive
            6. Sau khi huấn luyện xong, tải file `best.pt` về máy tính
            7. Đổi tên thành `animal_detection.pt` và đặt vào thư mục `models/`
            
            Quá trình huấn luyện có thể mất từ 2-4 giờ tùy thuộc vào GPU được cấp phát.
            Script đã được tối ưu để tự động lưu checkpoint và khôi phục khi bị crash.
            """)
            
        with st.expander("Hướng dẫn sử dụng"):
            st.markdown("""
            #### Cách sử dụng tính năng nhận dạng động vật:
            
            1. Chọn tính năng "🐾 Nhận dạng động vật" từ menu bên trái
            2. Sử dụng bộ lọc loài động vật ở sidebar:
               - Chọn các loài bạn muốn hiển thị kết quả 
               - Có thể nhấn "Chọn tất cả" hoặc "Bỏ chọn tất cả"
            3. Nhấn "Browse files" để tải lên ảnh chứa động vật
            4. Hệ thống sẽ tự động phát hiện và phân loại động vật trong ảnh
            5. Kết quả hiển thị với các bounding box và nhãn tương ứng
            6. Dưới ảnh kết quả sẽ hiển thị thông tin chi tiết:
               - Tóm tắt các loài động vật được phát hiện
               - Chi tiết về mỗi động vật (loài và độ tin cậy)
            """)
            
        with st.expander("Các loài động vật được hỗ trợ"):
            st.markdown("""
            Mô hình đã được huấn luyện để nhận diện nhiều loài động vật phổ biến, bao gồm:
            
            - Chó (Dog) 🐕
            - Mèo (Cat) 🐈
            - Chim (Bird) 🐦
            - Ngựa (Horse) 🐎
            - Bò (Cow) 🐄
            - Voi (Elephant) 🐘
            - Gấu (Bear) 🐻
            - Ngựa vằn (Zebra) 🦓
            - Hươu cao cổ (Giraffe) 🦒
            - Hổ (Tiger) 🐅
            
            Mỗi loài được hiển thị với emoji tương ứng trong kết quả nhận dạng.
            """)
            
        with st.expander("Mẹo để cải thiện kết quả"):
            st.markdown("""
            - **Ánh sáng**: Đảm bảo ảnh có đủ ánh sáng
            - **Góc nhìn**: Chụp động vật ở góc nhìn rõ ràng, không bị che khuất quá nhiều
            - **Khoảng cách**: Không chụp quá xa hoặc quá gần
            - **Chuyển động**: Tránh chụp khi động vật đang chuyển động nhanh
            - **Nhiều đối tượng**: Mô hình có thể nhận dạng nhiều động vật cùng lúc, nhưng tránh quá nhiều
              đối tượng chồng lấn nhau
            - **Nền đơn giản**: Cố gắng chụp với nền đơn giản để mô hình dễ dàng phân biệt động vật
            """)
    
    # Digital Image Processing
    elif selected_feature == "Xử lý ảnh số":
        st.markdown("### ⚙️ Hướng dẫn sử dụng tính năng Xử lý ảnh số")
        
        st.markdown("""
        Tính năng Xử lý ảnh số cung cấp nhiều phương pháp xử lý ảnh từ cơ bản đến nâng cao, 
        được tổ chức thành 3 chương chính theo nội dung môn học.
        """)
        
        with st.expander("Tổng quan các chương"):
            st.markdown("""
            Tính năng này chia thành 3 chương chính:
            
            1. **Chương 3 - Các phép toán điểm ảnh**: 
               - Xử lý ảnh ở mức pixel, áp dụng các biến đổi toán học cho từng điểm ảnh
               
            2. **Chương 4 - Xử lý tần số**:
               - Xử lý ảnh trong miền tần số sử dụng biến đổi Fourier
               
            3. **Chương 9 - Xử lý hình thái**:
               - Xử lý hình thái học với các phép toán co, giãn, biên, và đường bao
            """)
            
        with st.expander("Chương 3 - Các phép toán điểm ảnh", expanded=True):
            st.markdown("""
            ### Các phép toán điểm ảnh
            
            Chương này cung cấp các phép toán xử lý ảnh ở mức pixel:
            
            - **Negative**: Đảo âm bản ảnh (đảo ngược màu)
            - **NegativeColor**: Đảo âm bản ảnh màu
            - **Logarit**: Biến đổi logarit để tăng cường độ tương phản vùng tối
            - **Gamma**: Biến đổi gamma để điều chỉnh độ sáng và tương phản
            - **PiecewiseLine**: Biến đổi đoạn thẳng
            - **Histogram**: Hiển thị biểu đồ histogram của ảnh
            - **HistEqual**: Cân bằng histogram ảnh xám
            - **HistEqualColor**: Cân bằng histogram ảnh màu
            - **LocalHist**: Cân bằng histogram cục bộ
            - **HistStat**: Thống kê histogram
            - **SmoothBox**: Làm mịn ảnh với bộ lọc hộp
            - **Sharpening**: Làm sắc nét ảnh
            - **MySharpeningMask**: Làm sắc nét ảnh với mask
            - **SharpeningMask**: Làm sắc nét ảnh với mask Gaussian
            - **Gradient**: Phát hiện biên với gradient
            
            #### Khi nào sử dụng:
            
            - **Negative**: Khi cần đảo ngược màu để làm nổi bật chi tiết
            - **Logarit, Gamma**: Khi cần điều chỉnh độ tương phản cho ảnh quá tối hoặc quá sáng
            - **HistEqual**: Khi ảnh thiếu tương phản, các mức xám phân bố không đều
            - **Sharpening**: Khi ảnh bị mờ và cần làm nổi bật các cạnh, chi tiết
            - **Gradient**: Khi cần phát hiện biên của đối tượng
            """)
            
        with st.expander("Chương 4 - Xử lý tần số"):
            st.markdown("""
            ### Xử lý tần số
            
            Chương này cung cấp các phép toán xử lý ảnh trong miền tần số:
            
            - **Spectrum**: Hiển thị phổ tần số của ảnh
            - **RemoveMoire**: Loại bỏ hiệu ứng Moire thường thấy trong ảnh scan
            - **RemoveInterference**: Loại bỏ nhiễu giao thoa
            - **CreateMotion**: Tạo hiệu ứng chuyển động
            - **DeMotion**: Khử hiệu ứng chuyển động, khôi phục ảnh bị mờ do chuyển động
            
            #### Khi nào sử dụng:
            
            - **Spectrum**: Khi cần phân tích các thành phần tần số trong ảnh
            - **RemoveMoire**: Khi ảnh scan có hiệu ứng Moire (các mẫu lặp lại gây nhiễu)
            - **RemoveInterference**: Khi ảnh có nhiễu giao thoa từ nguồn điện từ
            - **DeMotion**: Khi ảnh bị mờ do chuyển động của máy ảnh hoặc đối tượng
            """)
            
        with st.expander("Chương 9 - Xử lý hình thái"):
            st.markdown("""
            ### Xử lý hình thái
            
            Chương này cung cấp các phép toán xử lý hình thái học:
            
            - **Erosion**: Phép co ảnh, thu nhỏ đối tượng và loại bỏ chi tiết nhỏ
            - **Dilation**: Phép giãn ảnh, mở rộng đối tượng và lấp đầy các lỗ hổng nhỏ
            - **Boundary**: Phát hiện biên của đối tượng
            - **Contour**: Vẽ đường bao đối tượng
            
            #### Khi nào sử dụng:
            
            - **Erosion**: Khi cần làm mỏng đối tượng hoặc loại bỏ chi tiết nhỏ, nhiễu
            - **Dilation**: Khi cần làm dày đối tượng hoặc kết nối các thành phần bị đứt
            - **Boundary**: Khi cần phát hiện biên của đối tượng
            - **Contour**: Khi cần xác định và vẽ đường bao cho đối tượng
            """)
            
        with st.expander("Hướng dẫn sử dụng"):
            st.markdown("""
            #### Cách sử dụng tính năng Xử lý ảnh số:
            
            1. Chọn tính năng "⚙️ Xử lý ảnh số" từ menu bên trái
            2. Chọn một chương xử lý từ sidebar (3, 4 hoặc 9)
            3. Chọn một phép toán cụ thể trong chương đó
            4. Nhấn "Browse files" để tải lên ảnh cần xử lý
            5. Ảnh gốc và ảnh kết quả sau xử lý sẽ hiển thị song song
            6. Có thể tải xuống ảnh kết quả bằng cách nhấn "Tải xuống kết quả"
            7. Chọn định dạng tải xuống (JPG, PNG hoặc TIFF)
            """)
            
        with st.expander("Định dạng ảnh và kích thước"):
            st.markdown("""
            - **Định dạng đầu vào hỗ trợ**: JPG, JPEG, PNG, WEBP, JFIF, TIF, TIFF
            - **Định dạng đầu ra**: JPG, PNG, TIFF
            - **Kích thước ảnh**: Không giới hạn, nhưng ảnh quá lớn có thể làm chậm xử lý
            - **Ảnh xám/màu**: Tùy thuộc vào phép toán, một số chỉ hoạt động với ảnh xám
              (hệ thống sẽ tự động chuyển đổi nếu cần)
            """)
    
    # Face Registration
    elif selected_feature == "Đăng ký khuôn mặt mới":
        st.markdown("### ➕ Hướng dẫn đăng ký khuôn mặt mới")
        
        st.markdown("""
        Tính năng đăng ký khuôn mặt mới cho phép bạn thêm người mới vào cơ sở dữ liệu nhận dạng khuôn mặt.
        Để đạt được kết quả nhận dạng tốt nhất, mỗi người nên có khoảng 100 ảnh mẫu.
        """)
        
        with st.expander("Tổng quan quy trình", expanded=True):
            st.markdown("""
            ### Quy trình đăng ký khuôn mặt mới
            
            1. **Thu thập dữ liệu**: Chụp nhiều ảnh khuôn mặt từ các góc và điều kiện ánh sáng khác nhau
            2. **Phát hiện khuôn mặt**: Sử dụng YuNet để phát hiện và cắt vùng khuôn mặt từ ảnh
            3. **Căn chỉnh khuôn mặt**: Căn chỉnh khuôn mặt để chuẩn hóa vị trí
            4. **Trích xuất đặc trưng**: Sử dụng SFace để chuyển đổi ảnh khuôn mặt thành vector embedding 128 chiều
            5. **Cập nhật cơ sở dữ liệu**: Lưu vector đặc trưng cùng với tên người
            """)
            
        with st.expander("Chế độ tự động"):
            st.markdown("""
            ### Chế độ tự động (Khuyên dùng)
            
            Chế độ này tự động chụp và xử lý ảnh khuôn mặt:
            
            1. **Chuẩn bị**:
               - Đảm bảo webcam hoạt động tốt và được kết nối
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
            """)
            
        with st.expander("Chế độ thủ công"):
            st.markdown("""
            ### Chế độ thủ công
            
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
            """)
            
        with st.expander("Mẹo để có kết quả tốt nhất"):
            st.markdown("""
            ### Mẹo để có kết quả tốt nhất
            
            - **Ánh sáng**: Đảm bảo khuôn mặt được chiếu sáng đầy đủ và đều
            - **Biểu cảm**: Thay đổi biểu cảm nhẹ (mỉm cười, nghiêm túc) để tăng độ đa dạng
            - **Góc nhìn**: Di chuyển đầu nhẹ nhàng sang trái, phải, lên, xuống
            - **Phụ kiện**: Thử đeo/bỏ kính, thay đổi kiểu tóc nếu có thể
            - **Tránh chuyển động nhanh**: Di chuyển từ từ để tránh ảnh bị mờ
            - **Khoảng cách**: Giữ khoảng cách 0.5-1m từ camera
            - **Nền**: Nếu có thể, sử dụng nền đơn giản
            """)
            
        with st.expander("Quản lý cơ sở dữ liệu"):
            st.markdown("""
            ### Quản lý cơ sở dữ liệu
            
            Ứng dụng cung cấp các chức năng quản lý cơ sở dữ liệu khuôn mặt:
            
            - **Xây dựng lại Database hoàn toàn**: Tạo lại cơ sở dữ liệu từ các ảnh mẫu
            - **Kiểm tra Database hiện tại**: Xem danh sách người đã đăng ký và số lượng mẫu
            
            > **Lưu ý**: Cần khởi động lại ứng dụng sau khi cập nhật cơ sở dữ liệu để áp dụng thay đổi.
            """)
    
    # 3D KITTI
    elif selected_feature == "Nhận dạng 3D KITTI":
        st.markdown("### 🚗 Hướng dẫn sử dụng tính năng Nhận dạng 3D KITTI")
        
        st.markdown("""
        Tính năng Nhận dạng 3D KITTI sử dụng công nghệ PointPillars để phát hiện và phân loại đối tượng 3D
        từ dữ liệu LiDAR trong bộ dữ liệu KITTI, một bộ dữ liệu quan trọng trong lĩnh vực xe tự lái.
        """)
        
        with st.expander("Giới thiệu về KITTI và PointPillars", expanded=True):
            st.markdown("""
            ### KITTI Dataset
            
            KITTI là một trong những bộ dữ liệu quan trọng nhất trong lĩnh vực xe tự lái, được thu thập bởi 
            Karlsruhe Institute of Technology và Toyota Technological Institute tại Chicago. Bộ dữ liệu này bao gồm:
            
            - Dữ liệu LiDAR 3D từ cảm biến Velodyne
            - Hình ảnh màu từ camera độ phân giải cao
            - Thông số hiệu chuẩn (calibration) giữa các cảm biến
            - Nhãn đối tượng: xe hơi, người đi bộ, xe đạp, v.v.
            
            ### Kiến trúc PointPillars
            
            PointPillars là một kiến trúc hiệu quả để xử lý dữ liệu point cloud từ LiDAR:
            
            1. **Pillar Feature Extractor (PFE)**:
               - Chuyển đổi point cloud dạng thưa thớt thành các "cột" (pillars)
               - Trích xuất đặc trưng từ các điểm trong mỗi cột
               - Tạo biểu diễn dạng lưới 2D của không gian 3D
            
            2. **Region Proposal Network (RPN)**:
               - Sử dụng đặc trưng từ PFE để dự đoán vị trí và lớp của đối tượng
               - Tạo ra các bounding box 3D với thông tin về vị trí, kích thước, hướng
               - Tính điểm tin cậy cho mỗi dự đoán
            """)
            
        with st.expander("Chuẩn bị"):
            st.markdown("""
            ### Chuẩn bị trước khi sử dụng
            
            Trước khi sử dụng tính năng này, bạn cần:
            
            1. **Tải KITTI dataset**:
               ```bash
               python scripts/download_kitti_dataset.py --data_dir data/kitti
               ```
            
            2. **Cài đặt thư viện bổ sung**:
               ```bash
               pip install open3d pyquaternion plotly transformations
               ```
            
            3. **Chuẩn bị mô hình**:
               - Đảm bảo có hai file mô hình ONNX trong thư mục `models/`:
                 - `pfe.onnx`: Pillar Feature Extractor
                 - `rpn.onnx`: Region Proposal Network
               - Các mô hình này có thể được tạo bằng script `scripts/train_pointpillars.py`
            """)
            
        with st.expander("Hướng dẫn sử dụng"):
            st.markdown("""
            #### Cách sử dụng tính năng Nhận dạng 3D KITTI:
            
            1. Chọn tính năng "🚗 Nhận dạng 3D KITTI" từ menu bên trái
            2. Nhấn "Lấy mẫu ngẫu nhiên" để tải một mẫu dữ liệu KITTI ngẫu nhiên
            3. Xem dữ liệu được hiển thị:
               - **Ảnh gốc**: Hình ảnh từ camera
               - **Point Cloud (2D view)**: Dữ liệu LiDAR được chiếu lên mặt phẳng 2D
            4. Nhấn "Nhận dạng đối tượng 3D" để thực hiện phát hiện đối tượng
            5. Xem kết quả:
               - **Kết quả 2D**: Các bounding box trên ảnh 2D
               - **Kết quả 3D**: Hiển thị point cloud và các box 3D trong không gian 3D tương tác
               - **Thông tin đối tượng**: Chi tiết về các đối tượng được phát hiện
            """)
            
        with st.expander("Tương tác với kết quả 3D"):
            st.markdown("""
            ### Tương tác với kết quả 3D
            
            Kết quả 3D hiển thị dưới dạng biểu đồ tương tác mà bạn có thể:
            
            - **Xoay**: Kéo chuột để xoay cảnh 3D
            - **Thu phóng**: Cuộn chuột để phóng to/nhỏ
            - **Di chuyển**: Nhấn Shift + kéo chuột để di chuyển
            
            Các đối tượng được hiển thị với màu khác nhau:
            
            - **Xe hơi (Car)**: Màu xanh lá
            - **Người đi bộ (Pedestrian)**: Màu đỏ
            - **Xe đạp (Cyclist)**: Màu xanh dương
            """)
            
        with st.expander("Giải thích thông tin đối tượng"):
            st.markdown("""
            ### Giải thích thông tin đối tượng
            
            Mỗi đối tượng được hiển thị với các thông tin:
            
            - **Loại**: Car, Pedestrian, Cyclist
            - **Điểm tin cậy**: Mức độ tin cậy từ 0-1 (càng cao càng chính xác)
            - **Vị trí**: Tọa độ (x, y, z) trong không gian 3D
              - x: hướng trước-sau
              - y: hướng trái-phải
              - z: hướng lên-xuống
            - **Kích thước**: Chiều dài, rộng, cao của đối tượng
            - **Góc quay**: Hướng của đối tượng theo độ
            """)