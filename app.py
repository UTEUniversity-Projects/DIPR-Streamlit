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
            
            /* Section title */
            .section-title {
                font-size: 14px;
                font-weight: 600;
                color: #9e9e9e;
                margin-bottom: 8px;
                text-transform: uppercase;
            }
            
            /* Sidebar nav divider */
            .nav-divider {
                height: 1px;
                background-color: #444;
                margin: 15px 0;
            }
            </style>
        """, unsafe_allow_html=True)

# Load CSS
load_css()

# Get logo base64
logo_base64 = get_logo_base64()

# Initialize session states
if 'current_mode' not in st.session_state:
    st.session_state.current_mode = "🏠 Trang chủ"

if 'active_tab' not in st.session_state:
    st.session_state.active_tab = 0  # Default to first tab (Introduction)

# Define the welcome page with tabs
def show_welcome_page():
    # Create tabs for Giới thiệu and Hướng dẫn
    tabs = st.tabs(["🔍 Giới thiệu", "📋 Hướng dẫn"])
    
    # Always show the active tab first
    with tabs[st.session_state.active_tab]:
        if st.session_state.active_tab == 0:
            from modules import introduction
            introduction.show()
        else:
            from modules import tutorial
            tutorial.show()
    
    # Also load content for the other tab (will be hidden but ready)
    with tabs[1 - st.session_state.active_tab]:
        if st.session_state.active_tab == 1:
            from modules import introduction
            introduction.show()
        else:
            from modules import tutorial
            tutorial.show()

# Functions to handle button clicks
def set_mode_home_intro():
    st.session_state.current_mode = "🏠 Trang chủ"
    st.session_state.active_tab = 0
    
def set_mode_home_tutorial():
    st.session_state.current_mode = "🏠 Trang chủ"
    st.session_state.active_tab = 1

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
    
    # Add section titles
    st.markdown('<p class="section-title">Thông tin</p>', unsafe_allow_html=True)
    
    # Information section buttons with on_click handlers
    col1, col2 = st.columns(2)
    with col1:
        st.button(
            "ℹ️ Giới thiệu", 
            on_click=set_mode_home_intro,
            use_container_width=True,
            type="primary" if (st.session_state.current_mode == "🏠 Trang chủ" and st.session_state.active_tab == 0) else "secondary"
        )
    with col2:
        st.button(
            "📚 Hướng dẫn", 
            on_click=set_mode_home_tutorial,
            use_container_width=True,
            type="primary" if (st.session_state.current_mode == "🏠 Trang chủ" and st.session_state.active_tab == 1) else "secondary"
        )
    
    # Divider
    st.markdown('<div class="nav-divider"></div>', unsafe_allow_html=True)
    
    # Processing functions section
    st.markdown('<p class="section-title">Chức năng xử lý</p>', unsafe_allow_html=True)
    
    # Navigation options for processing functions
    # Find the index of the current mode
    options = [
        "🏠 Trang chủ",
        "🖼️ Nhận dạng khuôn mặt",
        "🍎 Nhận dạng trái cây",
        "🐾 Nhận dạng động vật",
        "⚙️ Xử lý ảnh số",
        "➕ Đăng ký khuôn mặt mới",
        "🚗 Nhận dạng 3D KITTI"
    ]
    
    default_index = 0
    for i, option in enumerate(options):
        if option == st.session_state.current_mode:
            default_index = i
            break
    
    mode = st.selectbox(
        "Chọn chức năng",
        options=options,
        index=default_index
    )
    
    # Update the mode if changed
    if mode != st.session_state.current_mode:
        st.session_state.current_mode = mode
        st.rerun()

# Dynamic header based on selection
header_map = {
    "🏠": "🏠 Trang chủ",
    "🖼️": "🖼️ Nhận dạng khuôn mặt",
    "🍎": "🍎 Nhận dạng trái cây",
    "🐾": "🐾 Nhận dạng động vật",
    "⚙️": "⚙️ Xử lý ảnh số",
    "➕": "➕ Đăng ký khuôn mặt mới",
    "🚗": "🚗 Nhận dạng 3D KITTI"
}

# Get the header based on the current mode
current_header = "Xử Lý Ảnh Số"  # Default header
for key, value in header_map.items():
    if st.session_state.current_mode.startswith(key):
        current_header = value
        break

# Display header
st.markdown(f"""
    <div class="header-divider">
        <h1 class="simple-header">{current_header}</h1>
    </div>
""", unsafe_allow_html=True)

# Display content based on current mode
if st.session_state.current_mode.startswith("🏠"):
    # Show default welcome page with tabs
    show_welcome_page()
    
elif st.session_state.current_mode.startswith("🖼️"):
    from modules import face_recognition
    face_recognition.show()
    
elif st.session_state.current_mode.startswith("🍎"):
    from modules import fruit_detection
    fruit_detection.show()
    
elif st.session_state.current_mode.startswith("🐾"):
    from modules import animal_detection
    animal_detection.show()
    
elif st.session_state.current_mode.startswith("⚙️"):
    from modules import image_processing
    image_processing.show()
    
elif st.session_state.current_mode.startswith("➕"):
    from modules import face_registration
    face_registration.show()
    
elif st.session_state.current_mode.startswith("🚗"):
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