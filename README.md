# DIPR-Streamlit

Ứng dụng web Xử lý Ảnh Số sử dụng Streamlit, OpenCV và deep learning cho các bài toán nhận dạng khuôn mặt, nhận dạng trái cây, và các phép toán xử lý ảnh.

## Tính năng

1. **Nhận dạng khuôn mặt** (tối đa 5 người)
   - Sử dụng 2 mô hình ONNX: YuNet (phát hiện) và SFace (nhận dạng)
   - Hỗ trợ chụp ảnh từ webcam để xây dựng database
   - Mỗi người chụp 100 ảnh mẫu

2. **Nhận dạng trái cây** 
   - Sử dụng YOLOv8 để nhận dạng 5 loại trái cây
   - Hướng dẫn huấn luyện trên Google Colab
   - Dataset trái cây có thể mở rộng

3. **Xử lý ảnh số**
   - Chương 3: Các phép toán điểm ảnh (Negative, Logarit, Gamma, Histogram, v.v.)
   - Chương 4: Xử lý tần số (Spectrum, RemoveMoire, DeMotion, v.v.)
   - Chương 9: Xử lý hình thái (Erosion, Dilation, Boundary, Contour)

## Cài đặt

### Yêu cầu hệ thống
- Python 3.7+
- Webcam (cho chức năng chụp ảnh)
- GPU (tùy chọn, cho huấn luyện YOLOv8)

### Cài đặt thư viện
```bash
pip install -r requirements.txt
```

### Tải mô hình
1. Tải mô hình face detection và recognition:
   - [face_detection_yunet_2023mar.onnx](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)
   - [face_recognition_sface_2021dec.onnx](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface)
   
2. Đặt chúng vào thư mục `models/`

3. Huấn luyện mô hình YOLOv8 (xem hướng dẫn bên dưới) hoặc sử dụng mô hình có sẵn

## Sử dụng

### Chạy ứng dụng
```bash
streamlit run app.py
```

### Xây dựng database khuôn mặt

#### Chụp ảnh mẫu
```bash
python scripts/register_faces.py --mode capture --person_name "Tên người"
```

#### Xây dựng database
```bash
python scripts/register_faces.py --mode build
```

### Huấn luyện YOLOv8 trên Google Colab

1. Mở Google Colab
2. Copy nội dung file `scripts/train_yolov8.py` vào Colab notebook
3. Đảm bảo chọn runtime GPU
4. Chuẩn bị dataset trái cây theo format:
   ```
   dataset/
   ├── train/
   │   ├── images/
   │   └── labels/
   ├── val/
   │   ├── images/
   │   └── labels/
   └── dataset.yaml
   ```
5. Chạy script trong Colab
6. Tải model đã huấn luyện về và đặt vào `models/fruit_detection.pt`

## Cấu trúc dự án

```
project/
├── app.py                  # Ứng dụng Streamlit chính
├── requirements.txt        # Dependencies
├── models/
│   ├── face_detection_yunet_2023mar.onnx
│   ├── face_recognition_sface_2021dec.onnx
│   └── fruit_detection.pt
├── data/
│   ├── faces/             # Ảnh mẫu khuôn mặt
│   └── db_embeddings.pkl  # Database nhận dạng
├── utils/
│   ├── face_utils.py      # Xử lý khuôn mặt
│   ├── fruit_detection.py # Nhận dạng trái cây
│   ├── image_processing.py# Xử lý ảnh
│   └── chapter_utils.py   # Mapping chức năng
├── chapters/
│   ├── chapter3.py
│   ├── chapter4.py
│   └── chapter9.py
├── scripts/
│   ├── register_faces.py  # Đăng ký khuôn mặt
│   └── train_yolov8.py    # Huấn luyện YOLOv8
└── README.md
```

## Chi tiết chức năng

### 1. Nhận dạng khuôn mặt

- Sử dụng mô hình ONNX hiện đại cho độ chính xác cao
- Hỗ trợ nhận dạng đồng thời tối đa 5 người trong một khung hình
- Hiển thị tên và điểm tin cậy cho mỗi khuôn mặt được nhận dạng
- Database được xây dựng từ 100 ảnh mẫu mỗi người

### 2. Nhận dạng trái cây

- Hỗ trợ 5 loại trái cây: táo, chuối, cam, dâu tây, nho
- Sử dụng YOLOv8 cho độ chính xác cao
- Hiển thị bounding box và độ tin cậy
- Có thể tùy chỉnh dataset để nhận dạng thêm loại trái cây khác

### 3. Xử lý ảnh số

#### Chương 3 - Phép toán điểm ảnh:
- **Negative**: Đảo âm bản ảnh
- **Logarit**: Biến đổi logarit
- **Gamma**: Hiệu chỉnh gamma
- **Histogram**: Hiển thị và cân bằng histogram
- **Sharpening**: Làm sắc nét ảnh
- **Gradient**: Phát hiện biên

#### Chương 4 - Xử lý tần số:
- **Spectrum**: Hiển thị phổ tần số của ảnh
- **RemoveMoire**: Loại bỏ hiệu ứng Moire
- **RemoveInterference**: Loại bỏ nhiễu giao thoa
- **CreateMotion/DeMotion**: Tạo và khử hiệu ứng chuyển động

#### Chương 9 - Xử lý hình thái:
- **Erosion**: Phép co ảnh
- **Dilation**: Phép giãn ảnh
- **Boundary**: Phát hiện biên đối tượng
- **Contour**: Vẽ đường bao đối tượng

## Tips và troubleshooting

### Lỗi phổ biến

1. **Không phát hiện webcam**: Kiểm tra quyền truy cập camera
2. **Mô hình không load được**: Kiểm tra đường dẫn đến file .onnx hoặc .pt
3. **Lỗi xử lý ảnh**: Đảm bảo ảnh đầu vào đúng format

### Tối ưu hiệu suất

- Sử dụng GPU để huấn luyện YOLOv8 nhanh hơn
- Cache model trong Streamlit với `@st.cache_resource`
- Sử dụng ảnh có độ phân giải phù hợp

## Phát triển thêm

- Thêm chức năng chụp ảnh trực tiếp từ webcam trong ứng dụng
- Tích hợp thêm các mô hình deep learning khác
- Thêm chế độ batch processing cho nhiều ảnh
- Lưu lịch sử xử lý ảnh
- Export kết quả dưới nhiều định dạng

## Tác giả

Dự án được phát triển cho môn Xử lý Ảnh Số

## License

MIT License
