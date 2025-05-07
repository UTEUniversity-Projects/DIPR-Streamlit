import streamlit as st
import cv2
import numpy as np
import open3d as o3d
import os
import plotly.graph_objects as go
from typing import List, Dict, Tuple, Optional

# Import KITTIDetector từ utils
from utils.kitti_detection import KITTIDetector, prepare_kitti_sample

def plot_3d_visualization(vis_objects: List):
    """Tạo hiển thị 3D tương tác bằng Plotly"""
    fig = go.Figure()
    
    for obj in vis_objects:
        if isinstance(obj, o3d.geometry.PointCloud):
            # Thêm point cloud
            points = np.asarray(obj.points)
            colors = np.asarray(obj.colors) if obj.has_colors() else None
            
            # Lấy mẫu point cloud để hiển thị tốt hơn trên trình duyệt
            if len(points) > 5000:
                indices = np.random.choice(len(points), 5000, replace=False)
                points = points[indices]
                if colors is not None:
                    colors = colors[indices]
            
            # Tạo color array
            if colors is not None:
                # Chuyển đổi sang định dạng rgba cho Plotly
                color_vals = [f'rgb({int(r*255)},{int(g*255)},{int(b*255)})' for r, g, b in colors]
            else:
                # Sử dụng màu mặc định
                color_vals = 'rgb(100, 100, 255)'
                
            # Thêm point cloud vào figure
            fig.add_trace(go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode='markers',
                marker=dict(
                    size=1,
                    color=color_vals,
                    opacity=0.7
                ),
                name='Point Cloud'
            ))
        elif isinstance(obj, o3d.geometry.LineSet):
            # Thêm các đường của hộp giới hạn
            lines = np.asarray(obj.lines)
            points = np.asarray(obj.points)
            colors = np.asarray(obj.colors)
            
            for i, line in enumerate(lines):
                p1, p2 = points[line[0]], points[line[1]]
                
                # Lấy màu từ line set nếu có
                color_rgb = colors[i % len(colors)]
                
                # Chuyển đổi RGB float sang định dạng chuỗi cho Plotly
                color = f'rgb({int(color_rgb[0]*255)}, {int(color_rgb[1]*255)}, {int(color_rgb[2]*255)})'
                
                # Thêm đường vào figure
                fig.add_trace(go.Scatter3d(
                    x=[p1[0], p2[0]],
                    y=[p1[1], p2[1]],
                    z=[p1[2], p2[2]],
                    mode='lines',
                    line=dict(color=color, width=5),
                    showlegend=False
                ))
    
    # Cập nhật layout để hiển thị tốt hơn
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data',  # Giữ tỷ lệ thực tế
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.0)
            )
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig

def show():
    st.markdown("### Nhận dạng đối tượng 3D KITTI Dataset")
    
    # Kiểm tra file mô hình ONNX có tồn tại không
    pfe_path = "models/pfe.onnx"
    rpn_path = "models/rpn.onnx"
    
    # Hiển thị thông báo nếu không tìm thấy model
    if not os.path.exists(pfe_path) or not os.path.exists(rpn_path):
        st.error("Không tìm thấy file model ONNX")
        st.warning("""
        Cần hai file model:
        1. models/pfe.onnx - Pillar Feature Extractor
        2. models/rpn.onnx - Region Proposal Network
        
        Bạn có thể tạo các file này bằng cách chạy train_pointpillars.py hoặc tải về từ nguồn có sẵn.
        """)
        return
    
    # Kiểm tra dataset KITTI có tồn tại không
    data_dir = "data/kitti"
    if not os.path.exists(data_dir):
        st.error("Dataset KITTI chưa được tải về hoặc sai đường dẫn")
        st.info("""
        **Để sử dụng chức năng này, bạn cần:**
        1. Tải dataset KITTI bằng cách chạy:
           ```
           python scripts/download_kitti.py --data_dir data/kitti
           ```
        2. Đảm bảo bạn có hai file pfe.onnx và rpn.onnx trong thư mục models/
        3. Khởi động lại ứng dụng
        """)
        return
    
    try:
        # Tải detector
        with st.spinner("Đang tải mô hình PointPillars..."):
            detector = KITTIDetector(pfe_path=pfe_path, rpn_path=rpn_path)
            st.success("Đã tải model PointPillars thành công!")
    except Exception as e:
        st.error(f"Lỗi khi tải model: {str(e)}")
        st.warning("Không thể tải model PointPillars")
        st.info("Vui lòng kiểm tra lại file model và thư viện onnxruntime")
        return
    
    # Dataset browser
    st.markdown("### Duyệt KITTI Test set")
    
    # Tạo thanh bên để chọn các tùy chọn
    col1, col2 = st.columns([1, 1])
    
    with col1:
        sample_idx = st.number_input("Chọn ảnh mẫu", min_value=0, max_value=7480, value=0)
    
    with col2:
        st.markdown("&nbsp;")  # Khoảng trắng
        load_button = st.button("Tải ảnh mẫu", type="primary")
    
    # Xử lý khi nút Load được nhấn
    if load_button:
        try:
            with st.spinner("Đang tải dữ liệu mẫu..."):
                sample = prepare_kitti_sample(data_dir, sample_idx)
                
                # Lưu trong session state
                st.session_state['kitti_sample'] = sample
                st.success(f"Đã tải mẫu KITTI #{sample_idx}")
                
        except Exception as e:
            st.error(f"Lỗi khi tải ảnh mẫu: {str(e)}")
    
    # Hiển thị và xử lý dữ liệu KITTI
    if 'kitti_sample' in st.session_state:
        sample = st.session_state['kitti_sample']
        
        # Hiển thị dữ liệu gốc
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Ảnh gốc")
            st.image(cv2.cvtColor(sample['image'], cv2.COLOR_BGR2RGB), use_container_width=True)
        
        with col2:
            st.subheader("Point Cloud (2D view)")
            # Tạo một bản sao của ảnh gốc
            pcd_img = sample['image'].copy()
            
            # Hiển thị point cloud với màu dựa trên độ sâu
            try:
                # Lấy dữ liệu
                point_cloud = sample['point_cloud']
                calibs = sample['calibs']
                
                # Chiếu point cloud lên ảnh (dùng detector method)
                points_3d_cam = detector.project_lidar_to_camera(point_cloud[:, :3], calibs)
                
                # Lọc điểm nằm trước camera (z > 0)
                mask = points_3d_cam[:, 2] > 0
                points_3d_cam = points_3d_cam[mask]
                point_intensity = point_cloud[mask][:, 3] if point_cloud.shape[1] >= 4 else None
                
                # Chiếu sang ảnh 2D
                if 'P2' in calibs:
                    points_2d = detector.project_to_image(points_3d_cam, calibs, pcd_img.shape)
                    
                    # Lấy mẫu các điểm để giảm mật độ hiển thị (hiển thị khoảng 15% số điểm)
                    if len(points_2d) > 500:
                        sample_indices = np.random.choice(len(points_2d), size=max(500, len(points_2d) // 7), replace=False)
                        points_2d = points_2d[sample_indices]
                        depths = points_3d_cam[sample_indices][:, 2]
                    else:
                        depths = points_3d_cam[:, 2]
                    
                    # Tạo bản đồ màu dựa trên độ sâu
                    min_depth = np.min(depths)
                    max_depth = np.max(depths)
                    depth_range = max_depth - min_depth
                    
                    # Chỉ vẽ nếu có dữ liệu hợp lệ
                    if depth_range > 0:
                        # Tạo bản sao với độ trong suốt để hiển thị điểm
                        overlay = pcd_img.copy()
                        
                        # Vẽ điểm với màu và kích thước dựa trên độ sâu
                        for i, pt in enumerate(points_2d):
                            x, y = int(pt[0]), int(pt[1])
                            
                            # Tính màu dựa trên độ sâu (gần: đỏ -> xa: xanh)
                            depth_norm = (depths[i] - min_depth) / depth_range
                            
                            # Tạo màu từ colormap jet (đỏ -> vàng -> xanh lá -> xanh dương)
                            r = int(max(0, 255 * (1 - depth_norm * 2)) if depth_norm < 0.5 else 0)
                            g = int(max(0, 255 * (depth_norm * 2)) if depth_norm < 0.5 else max(0, 255 * (2 - depth_norm * 2)))
                            b = int(0 if depth_norm < 0.5 else max(0, 255 * (depth_norm * 2 - 1)))
                            
                            # Tính kích thước điểm dựa trên độ sâu (điểm gần lớn hơn)
                            point_size = max(1, int(3 * (1 - 0.7 * depth_norm)))
                            
                            # Vẽ điểm
                            cv2.circle(overlay, (x, y), point_size, (b, g, r), -1)
                        
                        # Kết hợp ảnh gốc và overlay với độ trong suốt
                        alpha = 0.7  # Độ đậm của point cloud
                        pcd_img = cv2.addWeighted(overlay, alpha, pcd_img, 1 - alpha, 0)
                        
                        # Thêm thanh màu để tham chiếu
                        h, w = pcd_img.shape[:2]
                        color_bar_width = w // 4
                        color_bar_height = 20
                        color_bar_x = w - color_bar_width - 10
                        color_bar_y = h - color_bar_height - 10
                        
                        # Vẽ thanh màu
                        for i in range(color_bar_width):
                            ratio = i / color_bar_width
                            r = int(max(0, 255 * (1 - ratio * 2)) if ratio < 0.5 else 0)
                            g = int(max(0, 255 * (ratio * 2)) if ratio < 0.5 else max(0, 255 * (2 - ratio * 2)))
                            b = int(0 if ratio < 0.5 else max(0, 255 * (ratio * 2 - 1)))
                            cv2.line(pcd_img, 
                                   (color_bar_x + i, color_bar_y), 
                                   (color_bar_x + i, color_bar_y + color_bar_height), 
                                   (b, g, r), 1)
                        
                        # Thêm nhãn
                        cv2.putText(pcd_img, f"{min_depth:.1f}m", 
                                   (color_bar_x, color_bar_y + color_bar_height + 15), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        cv2.putText(pcd_img, f"{max_depth:.1f}m", 
                                   (color_bar_x + color_bar_width - 40, color_bar_y + color_bar_height + 15), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        cv2.putText(pcd_img, "Depth", 
                                   (color_bar_x + color_bar_width // 2 - 20, color_bar_y - 5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Hiển thị ảnh với point cloud
                st.image(cv2.cvtColor(pcd_img, cv2.COLOR_BGR2RGB), use_container_width=True)
                
            except Exception as e:
                st.warning(f"Không thể hiển thị point cloud trên ảnh: {str(e)}")
                # Hiển thị point cloud dạng raw thay thế
                st.info("Hiển thị point cloud raw view (3D)")
                
                # Tạo point cloud open3d
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(sample['point_cloud'][:, :3])
                
                # Thêm màu dựa trên chiều cao
                colors = np.zeros((len(sample['point_cloud']), 3))
                colors[:, 0] = 0.8  # Default color (light blue)
                
                # Color by height
                min_z = np.min(sample['point_cloud'][:, 2])
                max_z = np.max(sample['point_cloud'][:, 2])
                if max_z > min_z:
                    height_colors = (sample['point_cloud'][:, 2] - min_z) / (max_z - min_z)
                    colors[:, 2] = height_colors  # Blue channel varies with height
                
                pcd.colors = o3d.utility.Vector3dVector(colors)
                
                # Hiển thị bằng Plotly
                fig = plot_3d_visualization([pcd])
                st.plotly_chart(fig, use_container_width=True)
        
        # Hiển thị nút phát hiện
        detect_button = st.button("Nhận dạng đối tượng 3D", type="primary")
        
        # Xử lý khi nút Detect được nhấn
        if detect_button:
            with st.spinner("Đang phân tích..."):
                try:
                    # Phát hiện đối tượng
                    detections, scores = detector.detect(sample['point_cloud'])
                    
                    if not detections:
                        st.warning("Không phát hiện đối tượng nào trong mẫu này")
                        return
                    
                    # Hiển thị kết quả 2D
                    st.subheader("Kết quả 2D")
                    result_2d = detector.visualize_2d(sample['image'].copy(), detections, scores, sample['calibs'])
                    st.image(cv2.cvtColor(result_2d, cv2.COLOR_BGR2RGB), use_container_width=True)
                    
                    # Hiển thị 3D
                    st.subheader("Kết quả 3D")
                    vis_objects = detector.visualize_3d(sample['point_cloud'], detections, scores)
                    
                    # Sử dụng Plotly cho hiển thị 3D tương tác
                    fig = plot_3d_visualization(vis_objects)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Tóm tắt phát hiện
                    st.subheader("Thông tin đối tượng")
                    
                    # Hiển thị thông tin chi tiết
                    with st.expander("Chi tiết đối tượng đã phát hiện", expanded=True):
                        if len(detections) > 0:
                            # Tạo nhiều cột để hiển thị thông tin
                            num_cols = min(3, len(detections))
                            cols = st.columns(num_cols)
                            
                            for i, (detection, score) in enumerate(zip(detections, scores)):
                                with cols[i % num_cols]:
                                    st.markdown(f"**Đối tượng {i+1}: {detection['class']}**")
                                    st.markdown(f"- **Điểm tin cậy:** {score:.2f}")
                                    st.markdown(f"- **Vị trí:** x={detection['location'][0]:.1f}, y={detection['location'][1]:.1f}, z={detection['location'][2]:.1f}")
                                    st.markdown(f"- **Kích thước:** {detection['dimensions'][0]:.1f} x {detection['dimensions'][1]:.1f} x {detection['dimensions'][2]:.1f}")
                                    
                                    # Hiển thị góc quay (đổi từ radian sang độ)
                                    rotation_deg = np.degrees(detection['rotation_y'])
                                    st.markdown(f"- **Góc quay:** {rotation_deg:.1f}°")
                        else:
                            st.info("Không có đối tượng nào được phát hiện")
                except Exception as e:
                    st.error(f"Lỗi khi phát hiện đối tượng: {str(e)}")
        
    # Thêm hướng dẫn và tài liệu tham khảo
    with st.expander("Hướng dẫn sử dụng và tài liệu", expanded=False):
        st.markdown("""
        ### Hướng dẫn sử dụng
        
        **KITTI Dataset** là một bộ dữ liệu chuẩn trong lĩnh vực nhận dạng và phát hiện đối tượng 3D, chủ yếu dùng cho xe tự hành. Mô-đun này sử dụng mô hình PointPillars để phát hiện đối tượng 3D từ dữ liệu point cloud.
        
        #### Cách sử dụng:
        1. **Chọn ảnh mẫu**: Chọn số thứ tự mẫu từ dataset KITTI và nhấn "Tải ảnh mẫu"
        2. **Nhận dạng đối tượng**: Nhấn "Nhận dạng đối tượng 3D" để chạy phát hiện
        3. **Upload dữ liệu tùy chỉnh**: Bạn có thể tải lên file point cloud dạng .bin để phát hiện
        
        #### Giải thích kết quả:
        - **Kết quả 2D**: Hiển thị ảnh với các bounding box 2D và nhãn
        - **Kết quả 3D**: Hiển thị point cloud và bounding box 3D tương tác
        - **Thông tin đối tượng**: Chi tiết về các đối tượng phát hiện được
        
        #### Tài liệu tham khảo:
        - [KITTI Dataset](http://www.cvlibs.net/datasets/kitti/eval_object.php)
        - [PointPillars Paper](https://arxiv.org/abs/1812.05784)
        - [OpenCV Zoo](https://github.com/opencv/opencv_zoo)
        
        #### Cải thiện hiệu suất:
        - Sử dụng GPU nếu có (cần cài đặt CUDA và onnxruntime-gpu)
        - Giảm số lượng điểm hiển thị nếu bị chậm
        """)