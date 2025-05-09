import cv2
import numpy as np
from typing import Dict, List, Callable, Any, Tuple
import importlib
import sys
import os

# Ensure the chapters folder is in the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import the chapter modules
from chapters import chapter3, chapter4, chapter9

def get_chapter3_functions() -> Dict[str, Callable]:
    """Get functions from Chapter 3"""
    functions = {
        "Negative": chapter3.Negative,
        "NegativeColor": chapter3.NegativeColor,
        "Logarit": chapter3.Logarit,
        "Gamma": chapter3.Gamma,
        "PiecewiseLine": chapter3.PiecewiseLine,
        "Histogram": chapter3.Histogram,
        "HistEqual": chapter3.HistEqual,
        "HistEqualColor": chapter3.HistEqualColor,
        "LocalHist": chapter3.LocalHist,
        "HistStat": chapter3.HistStat,
        "SmoothBox": chapter3.SmoothBox,
        "Sharpening": chapter3.Sharpening,
        "MySharpeningMask": chapter3.MySharpeningMask,
        "SharpeningMask": chapter3.SharpeningMask,
        "Gradient": chapter3.Gradient
    }
    return functions

def get_chapter4_functions() -> Dict[str, Callable]:
    """Get functions from Chapter 4"""
    functions = {
        "FrequencyFiltering": chapter4.FrequencyFiltering,
        "Spectrum": chapter4.Spectrum,
        "RemoveMoire": chapter4.RemoveMoire,
        "RemoveInterference": chapter4.RemoveInterference,
        "CreateMotion": chapter4.CreateMotion,
        "DeMotion": chapter4.DeMotion
    }
    return functions

def get_chapter9_functions() -> Dict[str, Callable]:
    """Get functions from Chapter 9"""
    functions = {
        "Erosion": chapter9.Erosion,
        "Dilation": chapter9.Dilation,
        "Boundary": chapter9.Boundary,
        "Contour": chapter9.Contour
    }
    return functions

def get_all_functions() -> Dict[str, Dict[str, Callable]]:
    """Get all functions organized by chapter"""
    return {
        "Chapter 3 - Điểm Ảnh": get_chapter3_functions(),
        "Chapter 4 - Xử lý tần số": get_chapter4_functions(),
        "Chapter 9 - Xử lý hình thái": get_chapter9_functions()
    }

def process_image(img: np.ndarray, func_name: str, params: Dict[str, Any] = None) -> np.ndarray:
    """
    Process image using the specified function
    Args:
        img: Input image (grayscale or BGR)
        func_name: Function name
        params: Additional parameters for the function
    Returns:
        processed_img: Processed image
    """
    # Get all functions
    all_functions = {}
    for chapter_funcs in get_all_functions().values():
        all_functions.update(chapter_funcs)
    
    # Check if function exists
    if func_name not in all_functions:
        raise ValueError(f"Function {func_name} not found")
    
    # Get function
    func = all_functions[func_name]
    
    # Set default parameters if none provided
    if params is None:
        params = {}
    
    # Apply function based on its parameter requirements
    if func_name == "FrequencyFiltering":
        # This function requires a specific filter parameter
        if "filter" not in params:
            # Use default identity filter
            h, w = img.shape[:2]
            H = np.ones((h, w), np.complex64)
            H.imag = 0.0
            return func(img, H)
        else:
            return func(img, params["filter"])
    else:
        # For most functions, just pass the image
        return func(img)

def prepare_image_for_processing(img: np.ndarray, func_name: str) -> np.ndarray:
    """
    Prepare image for processing with specific function
    Args:
        img: Input image (expected in BGR for color images)
        func_name: Function name
    Returns:
        prepared_img: Prepared image for processing (guaranteed to be in correct format)
    """
    # Verify input image is valid
    if img is None or img.size == 0:
        raise ValueError("Invalid input image (None or empty)")
        
    # Determine if function requires color or grayscale input
    color_functions = ["NegativeColor", "HistEqualColor"]
    
    if func_name in color_functions:
        # These functions expect color images (BGR)
        if len(img.shape) == 2:
            # Convert grayscale to BGR
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif len(img.shape) == 3 and img.shape[2] == 3:
            # Already a BGR image, return as is
            return img
        elif len(img.shape) == 3 and img.shape[2] == 4:
            # RGBA image, convert to BGR
            return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        else:
            raise ValueError(f"Unsupported image format: shape {img.shape}")
    else:
        # These functions expect grayscale images
        if len(img.shape) == 2:
            # Already grayscale, return as is
            return img
        elif len(img.shape) == 3 and img.shape[2] >= 3:
            # Color image, convert to grayscale
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError(f"Unsupported image format: shape {img.shape}")

def get_function_info() -> Dict[str, List[Dict[str, str]]]:
    """
    Get function information organized by chapter
    Returns:
        info: Dictionary of function information by chapter
    """
    chapter3_info = [
        {"name": "Negative", "description": "Đảo âm bản ảnh xám"},
        {"name": "NegativeColor", "description": "Đảo âm bản ảnh màu"},
        {"name": "Logarit", "description": "Biến đổi logarit"},
        {"name": "Gamma", "description": "Biến đổi gamma"},
        {"name": "PiecewiseLine", "description": "Biến đổi đoạn thẳng"},
        {"name": "Histogram", "description": "Hiển thị histogram"},
        {"name": "HistEqual", "description": "Cân bằng histogram ảnh xám"},
        {"name": "HistEqualColor", "description": "Cân bằng histogram ảnh màu"},
        {"name": "LocalHist", "description": "Cân bằng histogram cục bộ"},
        {"name": "HistStat", "description": "Thống kê histogram"},
        {"name": "SmoothBox", "description": "Làm mịn ảnh với bộ lọc hộp"},
        {"name": "Sharpening", "description": "Làm sắc nét ảnh"},
        {"name": "MySharpeningMask", "description": "Làm sắc nét ảnh với mask"},
        {"name": "SharpeningMask", "description": "Làm sắc nét ảnh với mask Gaussian"},
        {"name": "Gradient", "description": "Phát hiện biên với gradient"}
    ]
    
    chapter4_info = [
        {"name": "Spectrum", "description": "Hiển thị phổ tần số"},
        {"name": "RemoveMoire", "description": "Loại bỏ hiệu ứng Moire"},
        {"name": "RemoveInterference", "description": "Loại bỏ nhiễu giao thoa"},
        {"name": "CreateMotion", "description": "Tạo hiệu ứng chuyển động"},
        {"name": "DeMotion", "description": "Khử hiệu ứng chuyển động"}
    ]
    
    chapter9_info = [
        {"name": "Erosion", "description": "Phép co ảnh"},
        {"name": "Dilation", "description": "Phép giãn ảnh"},
        {"name": "Boundary", "description": "Phát hiện biên"},
        {"name": "Contour", "description": "Vẽ đường bao đối tượng"}
    ]
    
    return {
        "Chapter 3 - Điểm Ảnh": chapter3_info,
        "Chapter 4 - Xử lý tần số": chapter4_info,
        "Chapter 9 - Xử lý hình thái": chapter9_info
    }