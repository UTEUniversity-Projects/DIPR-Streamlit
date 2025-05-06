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
    return KITTIDetector(pfe_path="models/pfe.onnx", rpn_path="models/rpn.onnx")

def plot_3d_visualization(vis_objects: List):
    """Create interactive 3D visualization using Plotly"""
    fig = go.Figure()
    
    for obj in vis_objects:
        if isinstance(obj, o3d.geometry.PointCloud):
            # Add point cloud
            points = np.asarray(obj.points)
            
            # Subsample points for better performance in browser
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
            # Add bounding box lines
            lines = np.asarray(obj.lines)
            points = np.asarray(obj.points)
            colors = np.asarray(obj.colors)
            
            for i, line in enumerate(lines):
                p1, p2 = points[line[0]], points[line[1]]
                
                # Get color from line set if available
                color_rgb = colors[i] if i < len(colors) else [1, 0, 0]
                
                # Convert RGB float to string format for Plotly
                color = f'rgb({int(color_rgb[0]*255)}, {int(color_rgb[1]*255)}, {int(color_rgb[2]*255)})'
                
                fig.add_trace(go.Scatter3d(
                    x=[p1[0], p2[0]],
                    y=[p1[1], p2[1]],
                    z=[p1[2], p2[2]],
                    mode='lines',
                    line=dict(color=color, width=5),
                    showlegend=False
                ))
    
    # Update layout for better viewing
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
    
    # Check if ONNX model files exist
    if not os.path.exists("models/pfe.onnx") or not os.path.exists("models/rpn.onnx"):
        st.error("Không tìm thấy file model ONNX")
        st.warning("""
        Cần hai file model:
        1. models/pfe.onnx - Pillar Feature Extractor
        2. models/rpn.onnx - Region Proposal Network
        
        Vui lòng tải về từ: https://github.com/k0suke-murakami/kitti_pretrained_point_pillars
        """)
        return
    
    # Check if KITTI dataset exists
    data_dir = "data/kitti"
    if not os.path.exists(os.path.join(data_dir, "raw", "training")):
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
        # Load detector with ONNX models
        detector = load_kitti_detector()
        has_detector = True
        st.success("Đã tải model ONNX PointPillars thành công!")
    except Exception as e:
        st.error(f"Lỗi khi tải model: {str(e)}")
        has_detector = False
    
    if not has_detector:
        st.warning("Không thể tải model PointPillars")
        st.info("Vui lòng kiểm tra lại file model và thư viện onnxruntime")
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
            try:
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
            except Exception as e:
                st.warning(f"Không thể hiển thị point cloud trên ảnh: {str(e)}")
                # Display raw point cloud as alternative
                st.write("Point cloud raw view:")
                fig = go.Figure(data=[go.Scatter3d(
                    x=pcd[:1000, 0], 
                    y=pcd[:1000, 1], 
                    z=pcd[:1000, 2],
                    mode='markers',
                    marker=dict(
                        size=2,
                        color=pcd[:1000, 2],
                        colorscale='Viridis',
                    )
                )])
                st.plotly_chart(fig, use_container_width=True)
        
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