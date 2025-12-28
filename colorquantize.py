"""
ComfyUI Custom Node: Image Color Quantizer
==========================================
A preprocessing node that cleans up pixelated diagrams and graphics
by reducing colors to 2, 4, 8, or 16 dominant colors.

This removes pixelation artifacts and color distortions while
maintaining the original appearance of the image.
"""

import torch
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans, MiniBatchKMeans
from collections import Counter


class ImageColorQuantizer:
    """
    A ComfyUI node that reduces image colors to clean up pixelated graphics.
    
    Uses color quantization to find the dominant colors in an image and
    maps all pixels to their nearest dominant color, effectively removing
    noise and pixelation artifacts.
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "num_colors": (["2", "4", "8", "16"], {
                    "default": "8"
                }),
                "quantization_method": (["kmeans", "median_cut", "most_frequent", "kmeans_minibatch"], {
                    "default": "kmeans"
                }),
                "dithering": (["none", "floyd_steinberg"], {
                    "default": "none"
                }),
                "color_space": (["RGB", "LAB"], {
                    "default": "LAB"
                }),
            },
            "optional": {
                "sample_fraction": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.1,
                    "display": "slider"
                }),
                "smoothing_iterations": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 5,
                    "step": 1,
                    "display": "slider"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("quantized_image", "color_palette")
    FUNCTION = "quantize_colors"
    CATEGORY = "image/preprocessing"
    
    def quantize_colors(self, image, num_colors, quantization_method, 
                        dithering, color_space, sample_fraction=0.5, 
                        smoothing_iterations=0):
        """
        Main function to perform color quantization on the input image.
        
        Args:
            image: Input image tensor [B, H, W, C] with values 0-1
            num_colors: Number of colors to reduce to (2, 4, 8, or 16)
            quantization_method: Algorithm to use for finding dominant colors
            dithering: Whether to apply dithering (none or floyd_steinberg)
            color_space: Color space for quantization (RGB or LAB)
            sample_fraction: Fraction of pixels to sample for clustering
            smoothing_iterations: Number of smoothing passes to apply
            
        Returns:
            Tuple of (quantized_image, color_palette_visualization)
        """
        n_colors = int(num_colors)
        batch_results = []
        palette_results = []
        
        # Process each image in the batch
        for i in range(image.shape[0]):
            # Convert tensor to numpy array (0-255 range)
            img_np = (image[i].cpu().numpy() * 255).astype(np.uint8)
            
            # Get the dominant color palette
            if quantization_method == "kmeans":
                palette = self._kmeans_palette(img_np, n_colors, color_space, sample_fraction)
            elif quantization_method == "kmeans_minibatch":
                palette = self._kmeans_minibatch_palette(img_np, n_colors, color_space, sample_fraction)
            elif quantization_method == "median_cut":
                palette = self._median_cut_palette(img_np, n_colors)
            elif quantization_method == "most_frequent":
                palette = self._most_frequent_palette(img_np, n_colors)
            else:
                palette = self._kmeans_palette(img_np, n_colors, color_space, sample_fraction)
            
            # Apply quantization
            if dithering == "floyd_steinberg":
                quantized = self._apply_floyd_steinberg(img_np, palette)
            else:
                quantized = self._apply_nearest_color(img_np, palette, color_space)
            
            # Apply optional smoothing
            if smoothing_iterations > 0:
                quantized = self._apply_smoothing(quantized, palette, smoothing_iterations)
            
            # Convert back to tensor format (0-1 range)
            quantized_tensor = torch.from_numpy(quantized.astype(np.float32) / 255.0)
            batch_results.append(quantized_tensor)
            
            # Create palette visualization
            palette_vis = self._create_palette_visualization(palette, img_np.shape[1], img_np.shape[0])
            palette_tensor = torch.from_numpy(palette_vis.astype(np.float32) / 255.0)
            palette_results.append(palette_tensor)
        
        # Stack results back into batch
        output_image = torch.stack(batch_results, dim=0)
        output_palette = torch.stack(palette_results, dim=0)
        
        return (output_image, output_palette)
    
    def _rgb_to_lab(self, rgb):
        """Convert RGB to LAB color space for better perceptual distance."""
        # Normalize RGB values
        rgb_norm = rgb.astype(np.float32) / 255.0
        
        # Convert to XYZ
        mask = rgb_norm > 0.04045
        rgb_norm[mask] = ((rgb_norm[mask] + 0.055) / 1.055) ** 2.4
        rgb_norm[~mask] = rgb_norm[~mask] / 12.92
        rgb_norm *= 100
        
        # RGB to XYZ matrix
        x = rgb_norm[..., 0] * 0.4124564 + rgb_norm[..., 1] * 0.3575761 + rgb_norm[..., 2] * 0.1804375
        y = rgb_norm[..., 0] * 0.2126729 + rgb_norm[..., 1] * 0.7151522 + rgb_norm[..., 2] * 0.0721750
        z = rgb_norm[..., 0] * 0.0193339 + rgb_norm[..., 1] * 0.1191920 + rgb_norm[..., 2] * 0.9503041
        
        # Normalize for D65 white point
        x /= 95.047
        y /= 100.000
        z /= 108.883
        
        # XYZ to LAB
        epsilon = 0.008856
        kappa = 903.3
        
        fx = np.where(x > epsilon, x ** (1/3), (kappa * x + 16) / 116)
        fy = np.where(y > epsilon, y ** (1/3), (kappa * y + 16) / 116)
        fz = np.where(z > epsilon, z ** (1/3), (kappa * z + 16) / 116)
        
        L = 116 * fy - 16
        a = 500 * (fx - fy)
        b = 200 * (fy - fz)
        
        return np.stack([L, a, b], axis=-1)
    
    def _lab_to_rgb(self, lab):
        """Convert LAB back to RGB color space."""
        L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
        
        fy = (L + 16) / 116
        fx = a / 500 + fy
        fz = fy - b / 200
        
        epsilon = 0.008856
        kappa = 903.3
        
        xr = np.where(fx ** 3 > epsilon, fx ** 3, (116 * fx - 16) / kappa)
        yr = np.where(L > kappa * epsilon, ((L + 16) / 116) ** 3, L / kappa)
        zr = np.where(fz ** 3 > epsilon, fz ** 3, (116 * fz - 16) / kappa)
        
        # Multiply by reference white
        x = xr * 95.047
        y = yr * 100.000
        z = zr * 108.883
        
        # XYZ to RGB
        r = x * 3.2404542 / 100 + y * -1.5371385 / 100 + z * -0.4985314 / 100
        g = x * -0.9692660 / 100 + y * 1.8760108 / 100 + z * 0.0415560 / 100
        b_out = x * 0.0556434 / 100 + y * -0.2040259 / 100 + z * 1.0572252 / 100
        
        rgb = np.stack([r, g, b_out], axis=-1)
        
        # Apply gamma correction
        mask = rgb > 0.0031308
        rgb[mask] = 1.055 * (rgb[mask] ** (1/2.4)) - 0.055
        rgb[~mask] = 12.92 * rgb[~mask]
        
        rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
        return rgb
    
    def _kmeans_palette(self, img_np, n_colors, color_space, sample_fraction):
        """Extract dominant colors using K-means clustering."""
        h, w, c = img_np.shape
        pixels = img_np.reshape(-1, 3)
        
        # Sample pixels for faster processing
        n_samples = int(len(pixels) * sample_fraction)
        n_samples = max(n_samples, n_colors * 10)  # Ensure minimum samples
        indices = np.random.choice(len(pixels), min(n_samples, len(pixels)), replace=False)
        sampled_pixels = pixels[indices]
        
        # Convert to LAB if requested
        if color_space == "LAB":
            sampled_pixels = self._rgb_to_lab(sampled_pixels)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        kmeans.fit(sampled_pixels.astype(np.float32))
        
        palette = kmeans.cluster_centers_
        
        # Convert back to RGB if needed
        if color_space == "LAB":
            palette = self._lab_to_rgb(palette)
        else:
            palette = np.clip(palette, 0, 255).astype(np.uint8)
        
        return palette
    
    def _kmeans_minibatch_palette(self, img_np, n_colors, color_space, sample_fraction):
        """Extract dominant colors using Mini-Batch K-means (faster for large images)."""
        h, w, c = img_np.shape
        pixels = img_np.reshape(-1, 3)
        
        # Sample pixels
        n_samples = int(len(pixels) * sample_fraction)
        n_samples = max(n_samples, n_colors * 10)
        indices = np.random.choice(len(pixels), min(n_samples, len(pixels)), replace=False)
        sampled_pixels = pixels[indices]
        
        # Convert to LAB if requested
        if color_space == "LAB":
            sampled_pixels = self._rgb_to_lab(sampled_pixels)
        
        # Perform Mini-Batch K-means
        kmeans = MiniBatchKMeans(n_clusters=n_colors, random_state=42, batch_size=1024)
        kmeans.fit(sampled_pixels.astype(np.float32))
        
        palette = kmeans.cluster_centers_
        
        if color_space == "LAB":
            palette = self._lab_to_rgb(palette)
        else:
            palette = np.clip(palette, 0, 255).astype(np.uint8)
        
        return palette
    
    def _median_cut_palette(self, img_np, n_colors):
        """Extract dominant colors using the Median Cut algorithm."""
        # Use PIL's built-in quantization
        pil_img = Image.fromarray(img_np)
        quantized = pil_img.quantize(colors=n_colors, method=Image.Quantize.MEDIANCUT)
        
        # Extract palette
        palette_data = quantized.getpalette()[:n_colors * 3]
        palette = np.array(palette_data).reshape(-1, 3).astype(np.uint8)
        
        return palette
    
    def _most_frequent_palette(self, img_np, n_colors):
        """Extract the most frequently occurring colors."""
        h, w, c = img_np.shape
        pixels = img_np.reshape(-1, 3)
        
        # Reduce color precision to group similar colors
        reduced = (pixels // 8) * 8
        
        # Count color frequencies
        color_tuples = [tuple(c) for c in reduced]
        counter = Counter(color_tuples)
        
        # Get most common colors
        most_common = counter.most_common(n_colors)
        palette = np.array([list(c[0]) for c in most_common], dtype=np.uint8)
        
        # If we don't have enough unique colors, pad with existing
        while len(palette) < n_colors:
            palette = np.vstack([palette, palette[0:1]])
        
        return palette[:n_colors]
    
    def _apply_nearest_color(self, img_np, palette, color_space):
        """Map each pixel to its nearest color in the palette."""
        h, w, c = img_np.shape
        pixels = img_np.reshape(-1, 3)
        
        if color_space == "LAB":
            # Convert both to LAB for comparison
            pixels_lab = self._rgb_to_lab(pixels)
            palette_lab = self._rgb_to_lab(palette)
            
            # Find nearest color for each pixel
            distances = np.zeros((len(pixels), len(palette_lab)))
            for i, color in enumerate(palette_lab):
                distances[:, i] = np.sqrt(np.sum((pixels_lab - color) ** 2, axis=1))
        else:
            # Use RGB distance
            distances = np.zeros((len(pixels), len(palette)))
            for i, color in enumerate(palette):
                distances[:, i] = np.sqrt(np.sum((pixels.astype(np.float32) - color.astype(np.float32)) ** 2, axis=1))
        
        nearest_indices = np.argmin(distances, axis=1)
        quantized_pixels = palette[nearest_indices]
        
        return quantized_pixels.reshape(h, w, 3)
    
    def _apply_floyd_steinberg(self, img_np, palette):
        """Apply Floyd-Steinberg dithering with the given palette."""
        img_float = img_np.astype(np.float32)
        h, w, c = img_float.shape
        output = np.zeros_like(img_np)
        
        for y in range(h):
            for x in range(w):
                old_pixel = img_float[y, x].copy()
                
                # Find nearest palette color
                distances = np.sqrt(np.sum((palette.astype(np.float32) - old_pixel) ** 2, axis=1))
                nearest_idx = np.argmin(distances)
                new_pixel = palette[nearest_idx].astype(np.float32)
                
                output[y, x] = palette[nearest_idx]
                error = old_pixel - new_pixel
                
                # Distribute error to neighboring pixels
                if x + 1 < w:
                    img_float[y, x + 1] += error * 7 / 16
                if y + 1 < h:
                    if x > 0:
                        img_float[y + 1, x - 1] += error * 3 / 16
                    img_float[y + 1, x] += error * 5 / 16
                    if x + 1 < w:
                        img_float[y + 1, x + 1] += error * 1 / 16
        
        return output
    
    def _apply_smoothing(self, img_np, palette, iterations):
        """Apply smoothing by reassigning pixels based on their neighborhood."""
        result = img_np.copy()
        h, w, c = result.shape
        
        for _ in range(iterations):
            new_result = result.copy()
            
            for y in range(1, h - 1):
                for x in range(1, w - 1):
                    # Get 3x3 neighborhood
                    neighborhood = result[y-1:y+2, x-1:x+2].reshape(-1, 3)
                    
                    # Find most common color in neighborhood
                    color_tuples = [tuple(c) for c in neighborhood]
                    counter = Counter(color_tuples)
                    most_common = counter.most_common(1)[0][0]
                    
                    new_result[y, x] = most_common
            
            result = new_result
        
        return result
    
    def _create_palette_visualization(self, palette, width, height):
        """Create a visualization of the color palette."""
        n_colors = len(palette)
        swatch_width = width // n_colors
        
        vis = np.zeros((min(50, height // 4), width, 3), dtype=np.uint8)
        
        for i, color in enumerate(palette):
            start_x = i * swatch_width
            end_x = (i + 1) * swatch_width if i < n_colors - 1 else width
            vis[:, start_x:end_x] = color
        
        return vis


class ImageColorQuantizerAdvanced:
    """
    Advanced version with additional options for fine-tuning the quantization.
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "num_colors": ("INT", {
                    "default": 8,
                    "min": 2,
                    "max": 256,
                    "step": 1
                }),
                "quantization_method": (["kmeans", "median_cut", "octree", "most_frequent"], {
                    "default": "kmeans"
                }),
            },
            "optional": {
                "preserve_colors": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "#FFFFFF,#000000 (hex colors to preserve)"
                }),
                "background_color": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "#FFFFFF (optional background)"
                }),
                "edge_enhancement": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("quantized_image",)
    FUNCTION = "quantize_advanced"
    CATEGORY = "image/preprocessing"
    
    def quantize_advanced(self, image, num_colors, quantization_method,
                          preserve_colors="", background_color="", edge_enhancement=0.0):
        """
        Advanced quantization with additional features.
        """
        batch_results = []
        
        for i in range(image.shape[0]):
            img_np = (image[i].cpu().numpy() * 255).astype(np.uint8)
            
            # Parse preserved colors
            preserved = []
            if preserve_colors.strip():
                for color_str in preserve_colors.split(","):
                    color_str = color_str.strip()
                    if color_str.startswith("#") and len(color_str) == 7:
                        r = int(color_str[1:3], 16)
                        g = int(color_str[3:5], 16)
                        b = int(color_str[5:7], 16)
                        preserved.append([r, g, b])
            
            # Apply edge enhancement if requested
            if edge_enhancement > 0:
                img_np = self._enhance_edges(img_np, edge_enhancement)
            
            # Quantize using PIL (simple and reliable)
            pil_img = Image.fromarray(img_np)
            
            # Adjust number of colors if preserving some
            effective_colors = max(2, num_colors - len(preserved))
            
            if quantization_method == "octree":
                quantized = pil_img.quantize(colors=effective_colors, method=Image.Quantize.FASTOCTREE)
            elif quantization_method == "median_cut":
                quantized = pil_img.quantize(colors=effective_colors, method=Image.Quantize.MEDIANCUT)
            else:
                quantized = pil_img.quantize(colors=effective_colors, method=Image.Quantize.MAXCOVERAGE)
            
            result = np.array(quantized.convert("RGB"))
            
            # Add preserved colors back
            if preserved:
                result = self._add_preserved_colors(img_np, result, preserved)
            
            result_tensor = torch.from_numpy(result.astype(np.float32) / 255.0)
            batch_results.append(result_tensor)
        
        return (torch.stack(batch_results, dim=0),)
    
    def _enhance_edges(self, img_np, strength):
        """Enhance edges before quantization to preserve details."""
        from scipy import ndimage
        
        # Convert to grayscale for edge detection
        gray = np.mean(img_np.astype(np.float32), axis=2)
        
        # Sobel edge detection
        sobel_x = ndimage.sobel(gray, axis=1)
        sobel_y = ndimage.sobel(gray, axis=0)
        edges = np.sqrt(sobel_x**2 + sobel_y**2)
        edges = edges / edges.max() * strength
        
        # Enhance original image at edges
        enhanced = img_np.astype(np.float32)
        for c in range(3):
            enhanced[:, :, c] = np.clip(enhanced[:, :, c] * (1 + edges * 0.5), 0, 255)
        
        return enhanced.astype(np.uint8)
    
    def _add_preserved_colors(self, original, quantized, preserved_colors):
        """Map pixels close to preserved colors back to those exact colors."""
        result = quantized.copy()
        
        for preserved in preserved_colors:
            preserved = np.array(preserved, dtype=np.uint8)
            
            # Find pixels in original that are close to this preserved color
            distances = np.sqrt(np.sum((original.astype(np.float32) - preserved.astype(np.float32)) ** 2, axis=2))
            mask = distances < 30  # Threshold for "close" colors
            
            result[mask] = preserved
        
        return result