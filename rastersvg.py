"""
ComfyUI Custom Node: Raster to SVG Converter

This node converts raster diagram images with blurry/pixelated lines
into clean SVG format with uniform width lines.
"""

import numpy as np
import torch
import os
import folder_paths
from PIL import Image
from scipy import ndimage
from skimage import morphology, filters
from skimage.morphology import skeletonize
import svgwrite


class RasterToSVGConverter:
    """
    ComfyUI node that converts a raster diagram image to SVG format.
    Takes an image input and produces an SVG file with clean, uniform lines.
    """
    
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "line_width": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.5,
                    "max": 10.0,
                    "step": 0.5,
                    "display": "number"
                }),
                "filename_prefix": ("STRING", {
                    "default": "diagram"
                }),
            },
            "optional": {
                "threshold": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 255,
                    "step": 1,
                    "display": "number"
                }),
                "stroke_color": ("STRING", {
                    "default": "black"
                }),
                "invert_colors": ("BOOLEAN", {
                    "default": False
                }),
                "transparent_background": ("BOOLEAN", {
                    "default": True
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "IMAGE",)
    RETURN_NAMES = ("svg_string", "svg_path", "preview_image",)
    FUNCTION = "convert_to_svg"
    CATEGORY = "image/conversion"
    OUTPUT_NODE = True

    def convert_to_svg(self, image, line_width, filename_prefix, 
                   threshold=0, stroke_color="black", 
                   invert_colors=False, transparent_background=True):
        """
        Main conversion function called by ComfyUI.
        
        Args:
            image: Input image tensor from ComfyUI (B, H, W, C)
            line_width: Width of lines in the SVG output
            filename_prefix: Prefix for the output SVG filename
            threshold: Binary threshold (0 = auto using Otsu's method)
            stroke_color: Color of the strokes in SVG
            invert_colors: If True, treat white as lines instead of black
            transparent_background: If True, SVG has no background (default True)
        
        Returns:
            Tuple of (svg_path, preview_image)
        """
        # Convert from ComfyUI tensor format to numpy array
        # ComfyUI images are (B, H, W, C) with values 0-1
        img_tensor = image[0]  # Take first image in batch
        img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
        
        # Convert to grayscale if needed
        if len(img_np.shape) == 3 and img_np.shape[2] >= 3:
            # RGB to grayscale
            gray = np.dot(img_np[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
        else:
            gray = img_np
        
        image_size = gray.shape
        
        # Preprocess
        binary = self._preprocess(gray, threshold if threshold > 0 else None, invert_colors)
        
        # Extract skeleton
        skeleton = self._extract_skeleton(binary)
        
        # Trace paths
        paths = self._trace_paths(skeleton)
        
        # Generate SVG string
        svg_string = self._paths_to_svg(
            paths, 
            image_size, 
            line_width=line_width,
            stroke_color=stroke_color,
            transparent_background=transparent_background
        )
        
        # Save SVG to file
        svg_filename = f"{filename_prefix}_{self._get_counter():05d}.svg"
        svg_path = os.path.join(self.output_dir, svg_filename)
        with open(svg_path, 'w', encoding='utf-8') as f:
            f.write(svg_string)
        
        # Create preview image (render SVG paths back to image for preview)
        preview = self._create_preview(paths, image_size, line_width)
        
        # Convert preview to ComfyUI format (B, H, W, C)
        preview_tensor = torch.from_numpy(preview).unsqueeze(0).float() / 255.0
        
        return (svg_string, svg_path, preview_tensor,)
    
    def _get_counter(self):
        """Get a unique counter for filename."""
        counter_file = os.path.join(self.output_dir, ".svg_counter")
        if os.path.exists(counter_file):
            with open(counter_file, 'r') as f:
                counter = int(f.read().strip()) + 1
        else:
            counter = 1
        with open(counter_file, 'w') as f:
            f.write(str(counter))
        return counter
    
    def _preprocess(self, gray, threshold=None, invert=False):
        """Convert grayscale image to binary."""
        # Apply Gaussian blur to reduce noise
        smoothed = ndimage.gaussian_filter(gray, sigma=1)
        
        # Determine threshold
        if threshold is None:
            threshold = filters.threshold_otsu(smoothed)
        
        # Create binary image
        if invert:
            binary = smoothed > threshold  # White lines
        else:
            binary = smoothed < threshold  # Black lines
        
        # Clean up
        binary = morphology.remove_small_objects(binary, min_size=20)
        binary = morphology.remove_small_holes(binary, area_threshold=20)
        
        return binary
    
    def _extract_skeleton(self, binary):
        """Extract skeleton from binary image."""
        return skeletonize(binary)
    
    def _trace_paths(self, skeleton):
        """Trace all paths in the skeleton."""
        skeleton = skeleton.copy()
        paths = []
        
        # Find endpoints and junctions
        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        neighbor_count = ndimage.convolve(skeleton.astype(int), kernel, mode='constant')
        neighbor_count = neighbor_count * skeleton
        
        endpoints = list(zip(*np.where((neighbor_count == 1) & skeleton)))
        junctions = list(zip(*np.where((neighbor_count >= 3) & skeleton)))
        junction_set = set(junctions)
        
        neighbors_offset = [(-1, -1), (-1, 0), (-1, 1),
                           (0, -1),           (0, 1),
                           (1, -1),  (1, 0),  (1, 1)]
        
        def get_neighbors(y, x, skel):
            result = []
            for dy, dx in neighbors_offset:
                ny, nx = y + dy, x + dx
                if 0 <= ny < skel.shape[0] and 0 <= nx < skel.shape[1]:
                    if skel[ny, nx]:
                        result.append((ny, nx))
            return result
        
        def trace_from(start_y, start_x, skel, stop_at_junctions=True):
            path = [(start_x, start_y)]
            current = (start_y, start_x)
            skel[start_y, start_x] = False
            
            while True:
                neighbors = get_neighbors(current[0], current[1], skel)
                
                if not neighbors:
                    break
                
                junction_neighbors = [n for n in neighbors if n in junction_set]
                if junction_neighbors and stop_at_junctions:
                    next_point = junction_neighbors[0]
                    path.append((next_point[1], next_point[0]))
                    break
                
                next_point = neighbors[0]
                path.append((next_point[1], next_point[0]))
                skel[next_point[0], next_point[1]] = False
                current = next_point
            
            return path
        
        # Trace from endpoints
        for ey, ex in endpoints:
            if skeleton[ey, ex]:
                path = trace_from(ey, ex, skeleton)
                if len(path) > 1:
                    paths.append(path)
        
        # Trace remaining loops
        while True:
            remaining = np.where(skeleton)
            if len(remaining[0]) == 0:
                break
            
            start_y, start_x = remaining[0][0], remaining[1][0]
            path = trace_from(start_y, start_x, skeleton, stop_at_junctions=False)
            if len(path) > 1:
                paths.append(path)
        
        return paths
    
    
    def _smooth_path(self, path, window_size=5):
        """Smooth path using moving average."""
        if len(path) < window_size:
            return path
        
        path_array = np.array(path)
        smoothed = np.copy(path_array).astype(float)
        
        half_window = window_size // 2
        
        for i in range(half_window, len(path) - half_window):
            smoothed[i] = np.mean(path_array[i - half_window:i + half_window + 1], axis=0)
        
        return [tuple(p) for p in smoothed]
    
    def _paths_to_svg(self, paths, image_size, line_width=2, 
                    stroke_color='black', transparent_background=True):
        """Convert paths to SVG string.
        
        Returns:
            str: SVG content as a string
        """
        height, width = image_size
        
        dwg = svgwrite.Drawing(size=(f'{width}px', f'{height}px'),
                            viewBox=f'0 0 {width} {height}')
        
        if not transparent_background:
            dwg.add(dwg.rect(insert=(0, 0), size=(width, height), fill='white'))
        
        group = dwg.g(stroke=stroke_color, stroke_width=line_width,
                    fill='none', stroke_linecap='round', stroke_linejoin='round')
        
        for path in paths:
            if len(path) < 2:
                continue
            
            smoothed = self._smooth_path(path, window_size=3)
            
            # Build path with straight line segments (matching preview)
            path_data = f'M {smoothed[0][0]:.2f},{smoothed[0][1]:.2f}'
            for i in range(1, len(smoothed)):
                path_data += f' L {smoothed[i][0]:.2f},{smoothed[i][1]:.2f}'
            
            group.add(dwg.path(d=path_data))
        
        dwg.add(group)
        
        return dwg.tostring()
    
    def _create_preview(self, paths, image_size, line_width):
        """Create a preview image of the traced paths."""
        from PIL import Image, ImageDraw
        
        height, width = image_size
        preview = Image.new('RGB', (width, height), color=(255, 255, 255))
        draw = ImageDraw.Draw(preview)
        
        for path in paths:
            if len(path) < 2:
                continue
            
            smoothed = self._smooth_path(path, window_size=3)
            
            for i in range(len(smoothed) - 1):
                x1, y1 = smoothed[i]
                x2, y2 = smoothed[i + 1]
                draw.line([(x1, y1), (x2, y2)], fill=(0, 0, 0), width=int(line_width))
        
        return np.array(preview)