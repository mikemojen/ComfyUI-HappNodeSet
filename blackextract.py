"""
ComfyUI Custom Node: Diagram Cleanup
Converts any non-black pixels to pure white, isolating black line drawings.
"""

import torch
import numpy as np


class ExtractBlackColor:
    """
    A ComfyUI node that cleans up diagrams by converting any pixel 
    that isn't black into pure white RGB color.
    
    This is useful for isolating black line drawings from diagrams
    that may have colored backgrounds or other colored elements.
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "black_threshold": ("FLOAT", {
                    "default": 0.15,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider",
                    "tooltip": "Pixels with RGB values below this threshold are considered black (0.0 = pure black only, higher = more tolerance)"
                }),
            },
            "optional": {
                "invert_output": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "If enabled, outputs black background with white lines instead"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "cleanup_diagram"
    CATEGORY = "image/processing"
    DESCRIPTION = "Cleans up diagrams by converting non-black pixels to white, isolating black line drawings."

    def cleanup_diagram(self, image: torch.Tensor, black_threshold: float = 0.15, invert_output: bool = False) -> tuple[torch.Tensor]:
        """
        Process the image to isolate black lines.
        
        Args:
            image: Input image tensor in BHWC format with values 0-1
            black_threshold: Threshold below which pixels are considered black (0-1 range)
            invert_output: If True, invert the final output
            
        Returns:
            Tuple containing the processed image tensor
        """
        # Clone the input to avoid modifying the original
        result = image.clone()
        
        # Get the batch size
        batch_size = result.shape[0]
        
        for b in range(batch_size):
            # Get the current image (HWC format)
            img = result[b]
            
            # Handle both RGB and RGBA images
            if img.shape[-1] == 4:
                # RGBA: only process RGB channels, keep alpha
                rgb = img[:, :, :3]
                alpha = img[:, :, 3:4]
                has_alpha = True
            else:
                # RGB
                rgb = img
                has_alpha = False
            
            # Calculate the maximum RGB value for each pixel
            # A pixel is considered "black" if ALL its RGB values are below the threshold
            max_rgb = torch.max(rgb, dim=-1)[0]
            
            # Create a mask: True where pixel is considered black
            is_black = max_rgb < black_threshold
            
            # Create output image
            # Start with white (1.0) for all pixels
            output_rgb = torch.ones_like(rgb)
            
            # Keep original values where pixel is black
            # Expand the mask to match RGB channels
            is_black_expanded = is_black.unsqueeze(-1).expand_as(rgb)
            output_rgb = torch.where(is_black_expanded, rgb, output_rgb)
            
            # Optionally make black pixels pure black (0,0,0) for cleaner output
            output_rgb = torch.where(is_black_expanded, torch.zeros_like(rgb), output_rgb)
            
            # Invert if requested (black background, white lines)
            if invert_output:
                output_rgb = 1.0 - output_rgb
            
            # Reassemble with alpha if present
            if has_alpha:
                result[b] = torch.cat([output_rgb, alpha], dim=-1)
            else:
                result[b] = output_rgb
        
        return (result,)


class ExtractBlackColorAdvanced:
    """
    Advanced version with more control over the cleanup process.
    Allows separate thresholds for different aspects of "blackness".
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "luminance_threshold": ("FLOAT", {
                    "default": 0.2,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider",
                    "tooltip": "Pixels with luminance below this threshold are considered black"
                }),
                "saturation_tolerance": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider",
                    "tooltip": "Maximum saturation allowed for a pixel to be considered black (allows dark grays)"
                }),
            },
            "optional": {
                "output_mode": (["black_on_white", "white_on_black", "preserve_black_values"],),
                "apply_antialiasing": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Preserve anti-aliasing by keeping grayscale values for edge pixels"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "cleanup_diagram_advanced"
    CATEGORY = "image/processing"
    DESCRIPTION = "Advanced diagram cleanup with luminance-based detection and multiple output modes."

    def cleanup_diagram_advanced(
        self, 
        image: torch.Tensor, 
        luminance_threshold: float = 0.2,
        saturation_tolerance: float = 0.3,
        output_mode: str = "black_on_white",
        apply_antialiasing: bool = True
    ) -> tuple[torch.Tensor]:
        """
        Advanced diagram cleanup using luminance and saturation analysis.
        """
        result = image.clone()
        batch_size = result.shape[0]
        
        for b in range(batch_size):
            img = result[b]
            
            # Handle alpha channel
            if img.shape[-1] == 4:
                rgb = img[:, :, :3]
                alpha = img[:, :, 3:4]
                has_alpha = True
            else:
                rgb = img
                has_alpha = False
            
            # Calculate luminance (perceived brightness)
            # Using standard coefficients: 0.299*R + 0.587*G + 0.114*B
            luminance = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
            
            # Calculate saturation
            max_rgb = torch.max(rgb, dim=-1)[0]
            min_rgb = torch.min(rgb, dim=-1)[0]
            delta = max_rgb - min_rgb
            
            # Avoid division by zero
            saturation = torch.where(
                max_rgb > 0,
                delta / (max_rgb + 1e-7),
                torch.zeros_like(delta)
            )
            
            # A pixel is "black" if:
            # 1. Its luminance is below the threshold
            # 2. Its saturation is below the tolerance (it's not a dark colored pixel)
            is_black = (luminance < luminance_threshold) & (saturation < saturation_tolerance)
            
            # Create output based on mode
            if output_mode == "black_on_white":
                if apply_antialiasing:
                    # For anti-aliasing, use the luminance as grayscale for edge pixels
                    # Pixels that are somewhat dark but not fully black get intermediate values
                    edge_mask = (luminance < luminance_threshold * 2) & (~is_black)
                    
                    output_rgb = torch.ones_like(rgb)
                    is_black_expanded = is_black.unsqueeze(-1).expand_as(rgb)
                    edge_mask_expanded = edge_mask.unsqueeze(-1).expand_as(rgb)
                    luminance_expanded = luminance.unsqueeze(-1).expand_as(rgb)
                    
                    # Black pixels become pure black
                    output_rgb = torch.where(is_black_expanded, torch.zeros_like(rgb), output_rgb)
                    # Edge pixels get grayscale based on luminance
                    output_rgb = torch.where(edge_mask_expanded, luminance_expanded, output_rgb)
                else:
                    output_rgb = torch.ones_like(rgb)
                    is_black_expanded = is_black.unsqueeze(-1).expand_as(rgb)
                    output_rgb = torch.where(is_black_expanded, torch.zeros_like(rgb), output_rgb)
                    
            elif output_mode == "white_on_black":
                output_rgb = torch.zeros_like(rgb)
                is_black_expanded = is_black.unsqueeze(-1).expand_as(rgb)
                output_rgb = torch.where(is_black_expanded, torch.zeros_like(rgb), torch.ones_like(rgb))
                
            elif output_mode == "preserve_black_values":
                # Keep the original black pixel values (preserves anti-aliasing naturally)
                output_rgb = torch.ones_like(rgb)
                is_black_expanded = is_black.unsqueeze(-1).expand_as(rgb)
                output_rgb = torch.where(is_black_expanded, rgb, output_rgb)
            
            # Reassemble with alpha if present
            if has_alpha:
                result[b] = torch.cat([output_rgb, alpha], dim=-1)
            else:
                result[b] = output_rgb
        
        return (result,)