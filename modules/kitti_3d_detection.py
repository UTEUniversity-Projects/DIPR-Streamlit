import streamlit as st
import cv2
import numpy as np
import open3d as o3d
import os
import plotly.graph_objects as go
from typing import List, Dict, Tuple, Optional

# Quan trọng: Import KITTIDetector từ utils
from utils.kitti_detection import KITTIDetector, prepare_kitti_sample

def plot_3d_visualization(vis_objects: List):
    """Tạo hiển thị 3D tương tác bằng Plotly"""
    fig = go.Figure()
    
    for obj in vis_objects:
        if isinstance(obj, o3d.geometry.PointCloud):
            # Thêm point cloud
            points = np.asarray(obj.points)
            
            # Lấy mẫu point cloud để hiển thị tốt hơn trên trình duyệt
            if len(points) > 5000:
                indices = np.random.choice(len(points), 5000, replace=False)
                points = points[indices]
                
            fig.add_trace(go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode='markers',
                marker=dict(size=1, color='blue', opacity=0.5),
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
                color_rgb = colors[i] if i < len(colors) else [1, 0, 0]
                
                # Chuyển đổi RGB float sang định dạng chuỗi cho Plotly
                color = f'rgb({int(color_rgb[0]*255)}, {int(color_rgb[1]*255)}, {int(color_rgb[2]*255)})'
                
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
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=True
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
        
        Vui lòng tải về từ: https://github.com/k0suke-murakami/kitti_pretrained_point_pillars
        """)
        return
    
    # Kiểm tra dataset KITTI có tồn tại không
    data_dir = "data/kitti"
    if not os.path.exists(os.path.join(data_dir, "raw", "training")) and not os.path.exists(os.path.join(data_dir, "training")):
        st.error("Dataset KITTI chưa được tải về hoặc sai đường dẫn")
        st.info("""
        **Để sử dụng chức năng này, bạn cần:**
        1. Tải hai file pfe.onnx và rpn.onnx và đặt trong thư mục models/
        2. Tải dataset KITTI với cấu trúc như sau:
           data/kitti/
           └── raw/
               ├── training/
               │   ├── calib/
               │   ├── image_2/
               │   ├── label_2/
               │   └── velodyne/
               └── testing/
                   ├── calib/
                   ├── image_2/
                   └── velodyne/
        3. Khởi động lại ứng dụng
        """)
        return
    
    try:
        # Tải detector với CPU provider
        detector = KITTIDetector(pfe_path=pfe_path, rpn_path=rpn_path)
        has_detector = True
        st.success("Đã tải model ONNX PointPillars thành công!")
    except Exception as e:
        st.error(f"Lỗi khi tải model: {str(e)}")
        st.warning("Không thể tải model PointPillars")
        st.info("Vui lòng kiểm tra lại file model và thư viện onnxruntime")
        has_detector = False
        return
    
    # Dataset browser
    st.markdown("### Duyệt dataset KITTI")
    sample_idx = st.number_input("Chọn ảnh mẫu", min_value=0, max_value=7480, value=0)
    
    if st.button("Tải ảnh mẫu"):
        try:
            sample = prepare_kitti_sample(data_dir, sample_idx)
            
            # Lưu trong session state
            st.session_state['kitti_sample'] = sample
            
        except Exception as e:
            st.error(f"Lỗi khi tải ảnh mẫu: {str(e)}")
    
    # Xử lý và hiển thị
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
            
            # Sử dụng phương pháp đơn giản để hiển thị point cloud
            try:
                # Phương pháp mới - Không sử dụng project_camera_to_pixel
                point_cloud = sample['point_cloud']
                calibs = sample['calibs']
                image = sample['image']
                
                # Kiểm tra tọa độ calibration
                if 'Tr_velo_to_cam' not in calibs or calibs['Tr_velo_to_cam'] is None:
                    st.warning("Không tìm thấy ma trận biến đổi Velodyne sang Camera")
                    # Hiển thị point cloud dạng raw
                    st.write("Point cloud raw view:")
                    fig = go.Figure(data=[go.Scatter3d(
                        x=point_cloud[:1000, 0], 
                        y=point_cloud[:1000, 1], 
                        z=point_cloud[:1000, 2],
                        mode='markers',
                        marker=dict(
                            size=2,
                            color=point_cloud[:1000, 2],
                            colorscale='Viridis',
                        )
                    )])
                    st.plotly_chart(fig, use_container_width=True)
                    return
                
                # Phương pháp đơn giản:
                # 1. Chỉ lấy điểm XYZ (loại bỏ cường độ)
                pts_3d = point_cloud[:, :3]
                
                # 2. Thêm cột 1 để tạo tọa độ đồng nhất
                pts_3d_hom = np.hstack((pts_3d, np.ones((pts_3d.shape[0], 1))))
                
                # 3. Biến đổi từ không gian velodyne sang camera
                velo_to_cam = calibs['Tr_velo_to_cam']
                if velo_to_cam.shape == (3, 4):
                    # Mở rộng thành ma trận 4x4
                    temp = np.zeros((4, 4))
                    temp[:3, :4] = velo_to_cam
                    temp[3, 3] = 1
                    velo_to_cam = temp
                
                # Biến đổi điểm
                pts_cam = pts_3d_hom @ velo_to_cam.T
                
                # 4. Lọc điểm trước camera (z > 0)
                mask = pts_cam[:, 2] > 0
                pts_cam = pts_cam[mask]
                
                # 5. Chiếu từ camera sang pixel
                if 'P2' in calibs:
                    P = calibs['P2']
                    # Đảm bảo kích thước ma trận P đúng (3x4)
                    if P.shape != (3, 4):
                        if P.shape == (3, 3):
                            # Mở rộng P thành 3x4
                            P = np.hstack([P, np.zeros((3, 1))])
                        elif len(P.shape) == 1 and len(P) >= 12:
                            # Trường hợp P là vector, reshape thành 3x4
                            P = P[:12].reshape(3, 4)
                        else:
                            st.warning(f"Ma trận P2 có kích thước không hợp lệ: {P.shape}")
                            raise ValueError(f"Ma trận P2 có kích thước không hợp lệ: {P.shape}")
                    
                    # Chỉ lấy tọa độ x, y, z từ pts_cam (bỏ cột thứ 4)
                    pts_3d = pts_cam[:, :3]
                    
                    # Thêm cột 1 để tạo tọa độ đồng nhất (n,4)
                    pts_3d_hom = np.hstack((pts_3d, np.ones((pts_3d.shape[0], 1))))
                    
                    # Nhân với ma trận P
                    pts_2d_hom = pts_3d_hom @ P.T
                    
                    # Chuẩn hóa bằng cách chia cho tọa độ thứ 3
                    pts_2d = pts_2d_hom[:, :2] / pts_2d_hom[:, 2:3]
                    
                    # 6. Lọc điểm nằm trong ảnh
                    img_h, img_w = image.shape[:2]
                    mask = (pts_2d[:, 0] >= 0) & (pts_2d[:, 0] < img_w) & \
                           (pts_2d[:, 1] >= 0) & (pts_2d[:, 1] < img_h)
                    pts_2d = pts_2d[mask]
                    
                    # 7. Vẽ điểm
                    for pt in pts_2d:
                        x, y = int(pt[0]), int(pt[1])
                        cv2.circle(pcd_img, (x, y), 1, (0, 255, 0), -1)
                
                # Hiển thị ảnh với điểm
                st.image(cv2.cvtColor(pcd_img, cv2.COLOR_BGR2RGB), use_container_width=True)
                
            except Exception as e:
                st.warning(f"Không thể hiển thị point cloud trên ảnh: {str(e)}")
                # Hiển thị point cloud dạng raw thay thế
                st.write("Point cloud raw view:")
                fig = go.Figure(data=[go.Scatter3d(
                    x=sample['point_cloud'][:1000, 0], 
                    y=sample['point_cloud'][:1000, 1], 
                    z=sample['point_cloud'][:1000, 2],
                    mode='markers',
                    marker=dict(
                        size=2,
                        color=sample['point_cloud'][:1000, 2],
                        colorscale='Viridis',
                    )
                )])
                st.plotly_chart(fig, use_container_width=True)
        
        # Thực hiện phát hiện
        if st.button("Nhận dạng đối tượng 3D"):
            with st.spinner("Đang phân tích..."):
                # Phát hiện đối tượng
                detections, scores = detector.detect(sample['point_cloud'])
                
                # Hiển thị 2D
                st.subheader("Kết quả 2D")
                result_2d = detector.visualize_2d(sample['image'].copy(), detections, scores)
                st.image(cv2.cvtColor(result_2d, cv2.COLOR_BGR2RGB), use_container_width=True)
                
                # Hiển thị 3D
                st.subheader("Kết quả 3D")
                vis_objects = detector.visualize_3d(sample['point_cloud'], detections, scores)
                
                # Sử dụng Plotly cho hiển thị 3D tương tác
                fig = plot_3d_visualization(vis_objects)
                st.plotly_chart(fig, use_container_width=True)
                
                # Tóm tắt phát hiện
                st.subheader("Thông tin đối tượng")
                for i, (detection, score) in enumerate(zip(detections, scores)):
                    st.markdown(f"**Đối tượng {i+1}:**")
                    st.write(f"- Loại: {detection['class']}")
                    st.write(f"- Vị trí: x={detection['location'][0]:.1f}, y={detection['location'][1]:.1f}, z={detection['location'][2]:.1f}")
                    st.write(f"- Kích thước: {detection['dimensions'][0]:.1f} x {detection['dimensions'][1]:.1f} x {detection['dimensions'][2]:.1f}")
                    st.write(f"- Góc quay: {np.degrees(detection['rotation_y']):.1f}°")
                    st.write(f"- Độ tin cậy: {score:.2f}")
                    st.markdown("---")
                    
    # Upload custom data
    st.markdown("### Upload dữ liệu tùy chỉnh")
    
    # File upload for custom data
    point_cloud_file = st.file_uploader("Upload file Point Cloud (.bin)", type=["bin"])
    
    if point_cloud_file is not None:
        # Create temporary file for processing
        with open("temp_pointcloud.bin", "wb") as f:
            f.write(point_cloud_file.getbuffer())
        
        try:
            # Read point cloud
            point_cloud = detector.read_velodyne("temp_pointcloud.bin")
            
            # Simple visualization of point cloud
            st.write(f"Point cloud shape: {point_cloud.shape}")
            
            fig = go.Figure(data=[go.Scatter3d(
                x=point_cloud[:2000, 0], 
                y=point_cloud[:2000, 1], 
                z=point_cloud[:2000, 2],
                mode='markers',
                marker=dict(
                    size=2,
                    color=point_cloud[:2000, 2],
                    colorscale='Viridis',
                )
            )])
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Detection button
            if st.button("Nhận dạng đối tượng 3D trên dữ liệu upload"):
                with st.spinner("Đang phân tích..."):
                    # Detect objects
                    detections, scores = detector.detect(point_cloud)
                    
                    # 3D visualization
                    st.subheader("Kết quả 3D")
                    vis_objects = detector.visualize_3d(point_cloud, detections, scores)
                    
                    # Use Plotly for interactive 3D visualization
                    fig = plot_3d_visualization(vis_objects)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Detection summary
                    st.subheader("Thông tin đối tượng")
                    for i, (detection, score) in enumerate(zip(detections, scores)):
                        st.markdown(f"**Đối tượng {i+1}:**")
                        st.write(f"- Loại: {detection['class']}")
                        st.write(f"- Vị trí: x={detection['location'][0]:.1f}, y={detection['location'][1]:.1f}, z={detection['location'][2]:.1f}")
                        st.write(f"- Kích thước: {detection['dimensions'][0]:.1f} x {detection['dimensions'][1]:.1f} x {detection['dimensions'][2]:.1f}")
                        st.write(f"- Góc quay: {np.degrees(detection['rotation_y']):.1f}°")
                        st.write(f"- Độ tin cậy: {score:.2f}")
                        st.markdown("---")
        
        except Exception as e:
            st.error(f"Lỗi khi xử lý point cloud: {str(e)}")
            st.warning("Vui lòng kiểm tra lại file point cloud")
            
        # Clean up
        if os.path.exists("temp_pointcloud.bin"):
            os.remove("temp_pointcloud.bin")