import streamlit as st
import cv2
import numpy as np
import open3d as o3d
import os
import plotly.graph_objects as go
from typing import List, Dict

@st.cache_resource
def load_kitti_detector():
    from utils.kitti_detection import KITTIDetector
    return KITTIDetector("models/kitti_3d_detection.pt")

def plot_3d_visualization(vis_objects: List):
    """Create interactive 3D visualization using Plotly"""
    fig = go.Figure()
    
    for obj in vis_objects:
        if isinstance(obj, o3d.geometry.PointCloud):
            # Add point cloud
            points = np.asarray(obj.points)
            fig.add_trace(go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode='markers',
                marker=dict(size=1, color='blue', opacity=0.5),
                name='Point Cloud'
            ))
        elif isinstance(obj, o3d.geometry.LineSet):
            # Add bounding box lines
            lines = np.asarray(obj.lines)
            points = np.asarray(obj.points)
            
            for line in lines:
                p1, p2 = points[line[0]], points[line[1]]
                fig.add_trace(go.Scatter3d(
                    x=[p1[0], p2[0]],
                    y=[p1[1], p2[1]],
                    z=[p1[2], p2[2]],
                    mode='lines',
                    line=dict(color='red', width=5),
                    showlegend=False
                ))
    
    # Update layout
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
    
    # Check if KITTI dataset exists
    data_dir = "data/kitti"
    if not os.path.exists(os.path.join(data_dir, "raw")):
        st.error("Dataset KITTI chưa được tải về")
        st.info("""
        **Để sử dụng chức năng này, bạn cần:**
        1. Chạy script `python scripts/download_kitti.py` để tải dataset
        2. Huấn luyện model bằng script `python scripts/train_kitti_3d.py`
        3. Khởi động lại ứng dụng
        """)
        return
    
    try:
        detector = load_kitti_detector()
        has_detector = True
    except Exception:
        has_detector = False
    
    if not has_detector:
        st.warning("Model KITTI 3D chưa được huấn luyện")
        st.info("Vui lòng chạy script training trước khi sử dụng chức năng này")
        return
    
    # Dataset browser
    st.markdown("### Duyệt dataset KITTI")
    sample_idx = st.number_input("Chọn ảnh mẫu", min_value=0, max_value=7480, value=0)
    
    if st.button("Tải ảnh mẫu"):
        try:
            from utils.kitti_detection import prepare_kitti_sample
            sample = prepare_kitti_sample(data_dir, sample_idx)
            
            # Store in session state
            st.session_state['kitti_sample'] = sample
            
        except Exception as e:
            st.error(f"Lỗi khi tải ảnh mẫu: {str(e)}")
    
    # Process and visualize
    if 'kitti_sample' in st.session_state:
        sample = st.session_state['kitti_sample']
        
        # Display original data
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Ảnh gốc")
            st.image(cv2.cvtColor(sample['image'], cv2.COLOR_BGR2RGB), use_container_width=True)
        
        with col2:
            st.subheader("Point Cloud (2D view)")
            # Project point cloud to image
            pcd = sample['point_cloud'][:, :3]
            calibs = sample['calibs']
            
            # Transform from velodyne to camera coordinates
            Tr_velo_to_cam = np.hstack((calibs['Tr_velo_to_cam'], np.array([[0], [0], [0], [1]])))
            points_camera = (Tr_velo_to_cam @ np.vstack((pcd.T, np.ones((1, pcd.shape[0]))))).T
            
            # Filter points in front of camera
            in_front = points_camera[:, 2] > 0
            points_camera = points_camera[in_front]
            
            # Project to image
            points_2d = detector.project_camera_to_pixel(points_camera, calibs)
            
            # Draw point cloud on image
            pcd_img = sample['image'].copy()
            for point in points_2d:
                x, y = int(point[0]), int(point[1])
                if 0 <= x < pcd_img.shape[1] and 0 <= y < pcd_img.shape[0]:
                    cv2.circle(pcd_img, (x, y), 1, (0, 255, 0), -1)
            
            st.image(cv2.cvtColor(pcd_img, cv2.COLOR_BGR2RGB), use_container_width=True)
        
        # Perform detection
        if st.button("Nhận dạng đối tượng 3D"):
            with st.spinner("Đang phân tích..."):
                # Detect objects
                detections, scores = detector.detect(sample['point_cloud'])
                
                # 2D visualization
                st.subheader("Kết quả 2D")
                result_2d = detector.visualize_2d(sample['image'].copy(), detections, scores)
                st.image(cv2.cvtColor(result_2d, cv2.COLOR_BGR2RGB), use_container_width=True)
                
                # 3D visualization
                st.subheader("Kết quả 3D")
                vis_objects = detector.visualize_3d(sample['point_cloud'], detections, scores)
                
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
    
    # Upload custom data
    st.markdown("### Upload dữ liệu tùy chỉnh")
    st.info("Sắp tới: Hỗ trợ upload file Velodyne (.bin) và calibration tùy chỉnh")