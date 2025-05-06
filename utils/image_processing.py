import cv2
import numpy as np
from typing import Dict, List, Callable, Any
from . import chapter_utils

# Tích hợp các hàm xử lý ảnh từ các chương
class ImageProcessor:
    def __init__(self):
        self.functions = chapter_utils.get_all_functions()
    
    def get_available_functions(self) -> Dict[str, List[Dict[str, str]]]:
        """Get all available image processing functions with descriptions"""
        return chapter_utils.get_function_info()
    
    def process(self, img: np.ndarray, chapter: str, func_name: str) -> np.ndarray:
        """
        Process image using selected function from specific chapter
        Args:
            img: Input image
            chapter: Chapter number (3, 4, or 9)
            func_name: Function name
        Returns:
            Processed image
        """
        # Map chapter format to dictionary keys
        chapter_map = {
            "3": "Chapter 3 - Điểm Ảnh",
            "4": "Chapter 4 - Xử lý tần số", 
            "9": "Chapter 9 - Xử lý hình thái"
        }
        
        chapter_key = chapter_map.get(chapter)
        if not chapter_key or chapter_key not in self.functions:
            raise ValueError(f"Invalid chapter: {chapter}")
        
        if func_name not in self.functions[chapter_key]:
            raise ValueError(f"Function {func_name} not found in chapter {chapter}")
        
        # Prepare image for processing
        img_prepared = chapter_utils.prepare_image_for_processing(img, func_name)
        
        # Process image
        return chapter_utils.process_image(img_prepared, func_name)
    