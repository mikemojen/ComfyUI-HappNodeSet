import re
import numpy as np
from PIL import Image


class MaxValuePaddleOCR:
    """
    A ComfyUI node that uses PaddleOCR to extract numbers from an image
    and returns the largest numeric value found.
    """
    
    def __init__(self):
        self.ocr = None
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "use_gpu": ("BOOLEAN", {"default": False}),
                "lang": (["en", "ch", "japan", "korean", "french", "german"], {"default": "en"}),
            }
        }
    
    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("max_value",)
    FUNCTION = "extract_max_number"
    CATEGORY = "image/text"
    
    def _init_ocr(self, use_gpu=False, lang="en"):
        """Initialize PaddleOCR if not already done."""
        if self.ocr is None:
            from paddleocr import PaddleOCR
            self.ocr = PaddleOCR(
                use_angle_cls=True,
                lang=lang,
                use_gpu=use_gpu,
                show_log=False
            )
        return self.ocr
    
    def _extract_numbers(self, text):
        """
        Extract all numeric values (integers and floats) from text.
        Handles various number formats including negative numbers and decimals.
        """
        # Pattern matches integers and floats, including negative numbers
        # Examples: 123, -456, 3.14, -2.5, .5, -.5
        pattern = r'-?\d+\.?\d*|-?\.\d+'
        
        numbers = []
        matches = re.findall(pattern, text)
        
        for match in matches:
            try:
                # Try to convert to float (works for both int and float strings)
                num = float(match)
                numbers.append(num)
            except ValueError:
                continue
        
        return numbers
    
    def extract_max_number(self, image, use_gpu=False, lang="en"):
        """
        Main function to extract the maximum number from an image.
        
        Args:
            image: ComfyUI IMAGE tensor (B, H, W, C) in range [0, 1]
            use_gpu: Whether to use GPU for OCR
            lang: Language for OCR recognition
            
        Returns:
            Tuple containing the maximum numeric value found, or 0.0 if none found
        """
        # Initialize OCR
        ocr = self._init_ocr(use_gpu=use_gpu, lang=lang)
        
        # Convert ComfyUI image tensor to numpy array
        # ComfyUI images are (B, H, W, C) with values in [0, 1]
        if isinstance(image, np.ndarray):
            img_array = image
        else:
            # Convert from torch tensor if needed
            img_array = image.cpu().numpy()
        
        # Handle batch dimension - process first image
        if len(img_array.shape) == 4:
            img_array = img_array[0]
        
        # Convert from [0, 1] to [0, 255] and ensure uint8
        img_array = (img_array * 255).astype(np.uint8)
        
        # Convert to PIL Image then back to numpy for PaddleOCR
        # PaddleOCR expects BGR or RGB numpy array
        pil_image = Image.fromarray(img_array)
        img_for_ocr = np.array(pil_image)
        
        # Run OCR
        result = ocr.ocr(img_for_ocr, cls=True)
        
        # Extract all numbers from OCR results
        all_numbers = []
        
        if result and result[0]:
            for line in result[0]:
                if line and len(line) >= 2:
                    # line[1] contains (text, confidence)
                    text = line[1][0] if isinstance(line[1], (list, tuple)) else str(line[1])
                    numbers = self._extract_numbers(text)
                    all_numbers.extend(numbers)
        
        # Return the maximum value, or 0.0 if no numbers found
        if all_numbers:
            max_value = max(all_numbers)
        else:
            max_value = 0.0
        
        return (float(max_value),)