import streamlit as st
import base64
import os

# Configure page
st.set_page_config(
    page_title="Xử lý ảnh số", 
    layout="wide",
    page_icon="🖼️"
)

# Function to load and encode logo image
def get_logo_base64():
    """Load logo image and encode to base64"""
    logo_path = "assets/logo.png"
    if os.path.exists(logo_path):
        with open(logo_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    return None

# Load custom CSS
def load_css():
    """Load custom CSS from file"""
    css_path = "assets/styles.css"
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        # Fallback CSS if file doesn't exist
        st.markdown("""
            <style>
            /* Dark theme styling */
            .stApp {
                background-color: #1E1E1E;
            }
            
            /* Simple header styling */
            .simple-header {
                color: white;
                font-size: 36px;
                font-weight: bold;
                margin-bottom: 0;
                padding: 20px 0 10px 0;
            }
            
            .header-divider {
                border-bottom: 2px solid #444;
                margin-bottom: 30px;
                padding-bottom: 10px;
            }
            
            /* Sidebar styling */
            .sidebar .sidebar-content {
                background-color: #2B2B2B;
            }
            
            .sidebar-logo {
                width: 180px;
                height: 180px;
                margin: 5px auto 5px auto;
                display: block;
            }
            
            .main .block-container {
                padding-top: 2rem;
            }
            
            /* Make selectbox text visible on dark background */
            .stSelectbox label {
                color: white !important;
            }
            
            .stSelectbox > div > div {
                background-color: #3b3b3b;
                color: white;
            }
            
            /* Footer styling */
            .footer {
                text-align: center;
                color: white;
                margin-top: 50px;
                padding: 20px 0;
                border-top: 1px solid #444;
            }
            </style>
        """, unsafe_allow_html=True)

# Load CSS
load_css()

# Get logo base64
logo_base64 = get_logo_base64()

# Sidebar navigation with logo
with st.sidebar:
    # Display logo in sidebar  
    if logo_base64:
        st.markdown(f"""
            <img src="data:image/png;base64,{logo_base64}" class="sidebar-logo" alt="Logo">
        """, unsafe_allow_html=True)
    else:
        # Fallback logo if file doesn't exist
        st.markdown("""
            <div style="width: 180px; height: 180px; background-color: #4ECDC4; 
                        margin: 5px auto; display: flex; align-items: center; 
                        justify-content: center; color: white; font-size: 72px; 
                        font-weight: bold;">
                4
            </div>
        """, unsafe_allow_html=True)
    
    # Navigation options
    mode = st.selectbox(
        "Chọn chức năng",
        [
            "🖼️ Nhận dạng khuôn mặt",
            "🍎 Nhận dạng trái cây",
            "⚙️ Xử lý ảnh số",
            "➕ Đăng ký khuôn mặt mới",
            "🚗 Nhận dạng 3D KITTI"
        ]
    )

# Dynamic header based on selection
header_map = {
    "🖼️": "🖼️ Nhận dạng khuôn mặt",
    "🍎": "🍎 Nhận dạng trái cây",
    "⚙️": "⚙️ Xử lý ảnh số",
    "➕": "➕ Đăng ký khuôn mặt mới",
    "🚗": "🚗 Nhận dạng 3D KITTI"
}

# Get the header based on the selected mode
current_header = "Xử Lý Ảnh Số"  # Default header
for key, value in header_map.items():
    if mode.startswith(key):
        current_header = value
        break

st.markdown(f"""
    <div class="header-divider">
        <h1 class="simple-header">{current_header}</h1>
    </div>
""", unsafe_allow_html=True)

# Function selection based on mode
if mode.startswith("🖼️"):
    from modules import face_recognition
    face_recognition.show()
    
elif mode.startswith("🍎"):
    from modules import fruit_detection
    fruit_detection.show()
    
elif mode.startswith("⚙️"):
    from modules import image_processing
    image_processing.show()
    
elif mode.startswith("➕"):
    from modules import face_registration
    face_registration.show()
    
elif mode.startswith("🚗"):
    try:
        from modules import kitti_3d_detection
        kitti_3d_detection.show()
    except ImportError as e:
        st.error("Module KITTI 3D Detection chưa được cài đặt đầy đủ")
        st.error(str(e))
        st.info("""
        Để sử dụng tính năng này, hãy:
        1. Cài đặt các thư viện cần thiết
        2. Tải dataset KITTI
        3. Huấn luyện hoặc tải model đã huấn luyện
        """)

# Footer
st.markdown("""
<div class="footer">
    <p>KHOA CÔNG NGHỆ THÔNG TIN - HỌC PHẦN XỬ LÝ ẢNH SỐ</p>
    <p>Sinh viên Lê Hồng Phúc - 22110399</p>
</div>
""", unsafe_allow_html=True)

# Add JavaScript for dark mode
st.markdown("""
    <script>
    // Force dark mode
    window.addEventListener('load', function() {
        const root = document.documentElement;
        const theme = {
            colorText: "white",
            colorBackground: "#1E1E1E",
            colorPrimary: "#4ECDC4",
            colorTextSecondary: "#888888"
        };
        
        Object.entries(theme).forEach(([key, value]) => {
            root.style.setProperty(`--${key}`, value);
        });
    });
    </script>
""", unsafe_allow_html=True)