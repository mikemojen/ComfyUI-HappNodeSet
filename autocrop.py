"""
ComfyUI Custom Node: Auto Crop
Automatically crops white/transparent backgrounds from images, fitting the object to frame edges.
"""

import torch
import numpy as np


class AutoCropNode:
    """
    A ComfyUI node that automatically crops images by removing white or transparent
    backgrounds, fitting the object to the edge of the frame with a configurable margin.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "margin": ("INT", {
                    "default": 2,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "display": "number"
                }),
                "white_threshold": ("INT", {
                    "default": 250,
                    "min": 200,
                    "max": 255,
                    "step": 1,
                    "display": "number",
                    "tooltip": "Pixels with all RGB values above this are considered white (0-255)"
                }),
                "alpha_threshold": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "number",
                    "tooltip": "Pixels with alpha below this are considered transparent (0-1)"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("cropped_image",)
    FUNCTION = "auto_crop"
    CATEGORY = "image/transform"

    def auto_crop(self, image, margin=2, white_threshold=250, alpha_threshold=0.1):
        """
        Automatically crop the image to remove white/transparent backgrounds.
        
        Args:
            image: Input image tensor [B, H, W, C] with values in range 0-1
            margin: Pixel margin to add around the detected object
            white_threshold: RGB threshold for white detection (0-255 scale)
            alpha_threshold: Alpha threshold for transparency detection (0-1 scale)
            
        Returns:
            Tuple containing the cropped image tensor
        """
        # Convert threshold to 0-1 range
        white_thresh_normalized = white_threshold / 255.0
        
        # Process each image in the batch
        cropped_images = []
        
        for i in range(image.shape[0]):
            # Get single image from batch [H, W, C]
            img = image[i]
            
            # Convert to numpy for processing
            img_np = img.cpu().numpy()
            
            # Get image dimensions
            height, width, channels = img_np.shape
            
            # Create mask of non-background pixels
            if channels == 4:
                # RGBA image: check for both white and transparent pixels
                # A pixel is background if it's transparent OR white
                alpha = img_np[:, :, 3]
                rgb = img_np[:, :, :3]
                
                # Transparent pixels (alpha below threshold)
                is_transparent = alpha < alpha_threshold
                
                # White pixels (all RGB channels above threshold)
                is_white = np.all(rgb >= white_thresh_normalized, axis=2)
                
                # Background is transparent OR (opaque AND white)
                is_background = is_transparent | (is_white & (alpha >= alpha_threshold))
                
            else:
                # RGB image: only check for white pixels
                is_white = np.all(img_np[:, :, :3] >= white_thresh_normalized, axis=2)
                is_background = is_white
            
            # Object mask is the inverse of background
            object_mask = ~is_background
            
            # Find bounding box of the object
            rows_with_object = np.any(object_mask, axis=1)
            cols_with_object = np.any(object_mask, axis=0)
            
            # Check if there's any object detected
            if not np.any(rows_with_object) or not np.any(cols_with_object):
                # No object found, return original image
                cropped_images.append(img)
                continue
            
            # Get bounding box coordinates
            row_indices = np.where(rows_with_object)[0]
            col_indices = np.where(cols_with_object)[0]
            
            top = row_indices[0]
            bottom = row_indices[-1]
            left = col_indices[0]
            right = col_indices[-1]
            
            # Apply margin (with bounds checking)
            top = max(0, top - margin)
            bottom = min(height - 1, bottom + margin)
            left = max(0, left - margin)
            right = min(width - 1, right + margin)
            
            # Crop the image (bottom and right are inclusive, so add 1)
            cropped = img[top:bottom + 1, left:right + 1, :]
            cropped_images.append(cropped)
        
        # Handle batch with potentially different sizes
        # If all crops are the same size, stack them; otherwise, pad to largest size
        if len(cropped_images) == 1:
            result = cropped_images[0].unsqueeze(0)
        else:
            # Find maximum dimensions
            max_height = max(img.shape[0] for img in cropped_images)
            max_width = max(img.shape[1] for img in cropped_images)
            channels = cropped_images[0].shape[2]
            
            # Check if all images have the same size
            same_size = all(
                img.shape[0] == max_height and img.shape[1] == max_width 
                for img in cropped_images
            )
            
            if same_size:
                result = torch.stack(cropped_images, dim=0)
            else:
                # Pad images to the same size (pad with white/transparent)
                padded_images = []
                for img in cropped_images:
                    h, w, c = img.shape
                    if h < max_height or w < max_width:
                        # Create padded image (white background)
                        if c == 4:
                            # RGBA: transparent padding
                            padded = torch.zeros(max_height, max_width, c, 
                                                dtype=img.dtype, device=img.device)
                        else:
                            # RGB: white padding
                            padded = torch.ones(max_height, max_width, c, 
                                               dtype=img.dtype, device=img.device)
                        
                        # Center the cropped image in the padded area
                        y_offset = (max_height - h) // 2
                        x_offset = (max_width - w) // 2
                        padded[y_offset:y_offset + h, x_offset:x_offset + w, :] = img
                        padded_images.append(padded)
                    else:
                        padded_images.append(img)
                
                result = torch.stack(padded_images, dim=0)
        
        return (result,)