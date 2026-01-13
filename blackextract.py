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

class ExtractRedColor:
    """
    A ComfyUI node that extracts red colored elements from images by converting 
    any pixel that isn't red into pure white RGB color.
    
    This is useful for isolating red annotations, markings, or elements from 
    diagrams, documents, or images that may have other colored elements.
    
    A pixel is considered "red" when:
    - The red channel is above the minimum red threshold
    - The red channel dominates over green and blue by the specified amount
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "red_threshold": ("FLOAT", {
                    "default": 0.4,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider",
                    "tooltip": "Minimum red channel value required (0.0 = any red, 1.0 = pure red only)"
                }),
                "red_dominance": ("FLOAT", {
                    "default": 0.2,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider",
                    "tooltip": "How much the red channel must exceed green and blue (0.0 = no dominance required, higher = stricter)"
                }),
            },
            "optional": {
                "max_other_channels": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider",
                    "tooltip": "Maximum allowed value for green and blue channels (lower = purer reds only)"
                }),
                "preserve_original_red": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "If enabled, keeps original red pixel colors; if disabled, converts red pixels to pure red (1,0,0)"
                }),
                "invert_output": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "If enabled, outputs white background with inverted colors"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "extract_red"
    CATEGORY = "image/processing"
    DESCRIPTION = "Extracts red elements from images by converting non-red pixels to white, isolating red markings and annotations."

    def extract_red(
        self, 
        image: torch.Tensor, 
        red_threshold: float = 0.4, 
        red_dominance: float = 0.2,
        max_other_channels: float = 1.0,
        preserve_original_red: bool = True,
        invert_output: bool = False
    ) -> tuple[torch.Tensor]:
        """
        Process the image to isolate red pixels.
        
        Args:
            image: Input image tensor in BHWC format with values 0-1
            red_threshold: Minimum red channel value required (0-1 range)
            red_dominance: How much red must exceed max of green and blue (0-1 range)
            max_other_channels: Maximum allowed value for green and blue channels (0-1 range)
            preserve_original_red: If True, keep original red colors; if False, make them pure red
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
            
            # Extract individual channels
            r_channel = rgb[:, :, 0]
            g_channel = rgb[:, :, 1]
            b_channel = rgb[:, :, 2]
            
            # Calculate the maximum of green and blue channels
            max_gb = torch.max(g_channel, b_channel)
            
            # A pixel is considered "red" if:
            # 1. Red channel is above the threshold
            # 2. Red channel exceeds max(green, blue) by the dominance amount
            # 3. Green and blue are below the max_other_channels threshold
            is_red = (
                (r_channel >= red_threshold) & 
                (r_channel - max_gb >= red_dominance) &
                (max_gb <= max_other_channels)
            )
            
            # Create output image
            # Start with white (1.0) for all pixels
            output_rgb = torch.ones_like(rgb)
            
            # Expand the mask to match RGB channels
            is_red_expanded = is_red.unsqueeze(-1).expand_as(rgb)
            
            if preserve_original_red:
                # Keep original RGB values where pixel is red
                output_rgb = torch.where(is_red_expanded, rgb, output_rgb)
            else:
                # Convert red pixels to pure red (1, 0, 0)
                pure_red = torch.zeros_like(rgb)
                pure_red[:, :, 0] = 1.0  # Set red channel to 1
                output_rgb = torch.where(is_red_expanded, pure_red, output_rgb)
            
            # Invert if requested
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
    
"""
ComfyUI Custom Node: Non-White to Black Converter
Converts all non-white pixels in an image to solid black.
"""

import torch

class NonWhiteToBlack:
    """
    A ComfyUI node that converts all non-white pixels to solid black.
    White pixels (RGB: 255, 255, 255) remain white, all others become black.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "threshold": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.9,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider"
                }),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "convert_non_white_to_black"
    CATEGORY = "image/color"
    
    def convert_non_white_to_black(self, image: torch.Tensor, threshold: float = 1.0) -> tuple[torch.Tensor]:
        """
        Convert all non-white pixels to black.
        
        Args:
            image: Input image tensor with shape (B, H, W, C) where C is RGB(A)
                   Values are expected to be in range [0, 1]
            threshold: Threshold for considering a pixel as white (default 1.0 for pure white)
                      Lower values allow near-white pixels to remain white
        
        Returns:
            Tuple containing the processed image tensor
        """
        # Clone the image to avoid modifying the original
        result = image.clone()
        
        # Get the RGB channels (first 3 channels)
        rgb = result[..., :3]
        
        # Create a mask for white pixels
        # A pixel is considered white if ALL RGB channels are >= threshold
        white_mask = (rgb >= threshold).all(dim=-1, keepdim=True)
        
        # Expand mask to cover all RGB channels
        white_mask_rgb = white_mask.expand_as(rgb)
        
        # Set non-white pixels to black (0) and keep white pixels as white (1)
        result[..., :3] = torch.where(white_mask_rgb, rgb, torch.zeros_like(rgb))
        
        # If there's an alpha channel, preserve it
        # (result already contains the original alpha from the clone)
        
        return (result,)