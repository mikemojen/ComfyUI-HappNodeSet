"""
Line Detector Node for ComfyUI
Implements skeletonization-based line detection with junction analysis.

Outputs:
1. Number of lines detected
2. Comma-separated line lengths
3. Image with each line highlighted in a different color
"""

import numpy as np
from collections import defaultdict
import math
import torch
from PIL import Image
import cv2

class LineDetectorNode:
    """
    ComfyUI node that detects individual straight lines in diagram images.
    
    Rules:
    - 4-way junctions: Two lines crossing (paired by optimal collinearity)
    - All other junctions: Each connection is a separate segment endpoint
    """
    
    # Distinct colors for line visualization (RGB, 0-255)
    LINE_COLORS = [
        (255, 0, 0),      # Red
        (0, 255, 0),      # Green
        (0, 0, 255),      # Blue
        (255, 255, 0),    # Yellow
        (255, 0, 255),    # Magenta
        (0, 255, 255),    # Cyan
        (255, 128, 0),    # Orange
        (128, 0, 255),    # Purple
        (0, 255, 128),    # Spring Green
        (255, 0, 128),    # Rose
        (128, 255, 0),    # Lime
        (0, 128, 255),    # Sky Blue
        (255, 128, 128),  # Light Red
        (128, 255, 128),  # Light Green
        (128, 128, 255),  # Light Blue
        (255, 255, 128),  # Light Yellow
        (255, 128, 255),  # Light Magenta
        (128, 255, 255),  # Light Cyan
        (192, 64, 0),     # Brown
        (64, 192, 0),     # Dark Lime
    ]
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "threshold": ("INT", {
                    "default": 128,
                    "min": 0,
                    "max": 255,
                    "step": 1,
                    "display": "slider"
                }),
                "invert": ("BOOLEAN", {
                    "default": False,
                    "label_on": "Lines are white",
                    "label_off": "Lines are black"
                }),
                "min_line_length": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 1000,
                    "step": 1,
                    "display": "number"
                }),
                "line_thickness": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "display": "slider"
                }),
            }
        }
    
    RETURN_TYPES = ("INT", "STRING", "IMAGE",)
    RETURN_NAMES = ("line_count", "line_lengths", "colored_lines_image",)
    FUNCTION = "detect_lines"
    CATEGORY = "image/analysis"
    
    def detect_lines(self, image, threshold=128, invert=False, 
                     min_line_length=10, line_thickness=2):
        """
        Main function to detect and measure lines in the input image.
        
        Args:
            image: Input image tensor from ComfyUI (B, H, W, C)
            threshold: Binarization threshold (0-255)
            invert: If True, treat white pixels as lines; otherwise black pixels
            min_line_length: Minimum length to count as a valid line
            line_thickness: Thickness of lines in output visualization
            
        Returns:
            Tuple of (line_count, comma-separated line lengths, colored image)
        """
        import torch
        
        # Convert ComfyUI image tensor to numpy array
        img_np = self._tensor_to_numpy(image)
        height, width = img_np.shape[:2]
        
        # Preprocess: convert to binary
        binary = self._binarize(img_np, threshold, invert)
        
        # Skeletonize
        skeleton = self._skeletonize(binary)
        
        # Build graph from skeleton
        nodes, edges, node_positions = self._build_graph(skeleton)
        
        if not edges:
            # No lines detected - return empty results
            empty_image = np.ones((height, width, 3), dtype=np.float32)
            empty_tensor = torch.from_numpy(empty_image).unsqueeze(0)
            return (0, "No lines detected", empty_tensor)
        
        # Extract complete lines with junction rules
        lines, line_pixels = self._extract_complete_lines(nodes, edges, node_positions)
        
        # Calculate lengths and filter by minimum length
        valid_lines = []
        valid_pixels = []
        line_lengths = []
        
        for i, line_edges in enumerate(lines):
            length = 0
            for edge_id in line_edges:
                if edge_id in edges:
                    _, _, path = edges[edge_id]
                    for j in range(len(path) - 1):
                        r1, c1 = path[j]
                        r2, c2 = path[j + 1]
                        length += math.sqrt((r2 - r1) ** 2 + (c2 - c1) ** 2)
            
            if length >= min_line_length:
                valid_lines.append(line_edges)
                valid_pixels.append(line_pixels[i])
                line_lengths.append(length)
        
        # Generate colored visualization
        colored_image = self._create_colored_image(
            height, width, valid_pixels, line_thickness, binary
        )
        
        # Convert to ComfyUI tensor format
        colored_tensor = torch.from_numpy(colored_image).unsqueeze(0)
        
        # Format output
        line_count = len(line_lengths)
        lengths_str = ", ".join([f"{length:.2f}" for length in sorted(line_lengths, reverse=True)])
        
        if line_count == 0:
            lengths_str = "No lines detected"
        
        return (line_count, lengths_str, colored_tensor)
    
    def _tensor_to_numpy(self, tensor) -> np.ndarray:
        """Convert ComfyUI image tensor to numpy grayscale array."""
        # ComfyUI images are (B, H, W, C) with values 0-1
        img = tensor[0].cpu().numpy()  # Take first image in batch
        img = (img * 255).astype(np.uint8)
        
        # Convert to grayscale if needed
        if len(img.shape) == 3 and img.shape[2] >= 3:
            # Use luminosity method
            gray = (0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2])
            return gray.astype(np.uint8)
        return img
    
    def _binarize(self, img: np.ndarray, threshold: int, invert: bool) -> np.ndarray:
        """Convert grayscale image to binary."""
        if invert:
            return (img > threshold).astype(np.uint8)
        else:
            return (img < threshold).astype(np.uint8)
    
    def _skeletonize(self, binary: np.ndarray) -> np.ndarray:
        """
        Perform morphological skeletonization (Zhang-Suen thinning).
        Reduces all lines to single-pixel width while preserving topology.
        """
        skeleton = binary.copy()
        
        def get_neighbors(img, r, c):
            """Get 8-connected neighbors in order: P2,P3,P4,P5,P6,P7,P8,P9"""
            return [
                img[r-1, c],   # P2 (north)
                img[r-1, c+1], # P3 (northeast)
                img[r, c+1],   # P4 (east)
                img[r+1, c+1], # P5 (southeast)
                img[r+1, c],   # P6 (south)
                img[r+1, c-1], # P7 (southwest)
                img[r, c-1],   # P8 (west)
                img[r-1, c-1], # P9 (northwest)
            ]
        
        def transitions(neighbors):
            """Count 0->1 transitions in the ordered sequence."""
            n = neighbors + [neighbors[0]]  # Wrap around
            return sum(n[i] == 0 and n[i+1] == 1 for i in range(8))
        
        def step(img, step_num):
            """Perform one step of Zhang-Suen thinning."""
            rows, cols = img.shape
            to_remove = []
            
            for r in range(1, rows - 1):
                for c in range(1, cols - 1):
                    if img[r, c] != 1:
                        continue
                    
                    neighbors = get_neighbors(img, r, c)
                    n_count = sum(neighbors)
                    
                    if not (2 <= n_count <= 6):
                        continue
                    if transitions(neighbors) != 1:
                        continue
                    
                    P2, P3, P4, P5, P6, P7, P8, P9 = neighbors
                    
                    if step_num == 1:
                        if P2 * P4 * P6 != 0:
                            continue
                        if P4 * P6 * P8 != 0:
                            continue
                    else:
                        if P2 * P4 * P8 != 0:
                            continue
                        if P2 * P6 * P8 != 0:
                            continue
                    
                    to_remove.append((r, c))
            
            for r, c in to_remove:
                img[r, c] = 0
            
            return len(to_remove) > 0
        
        # Pad image to handle borders
        padded = np.pad(skeleton, 1, mode='constant', constant_values=0)
        
        # Iterate until no more changes
        changed = True
        max_iterations = 1000
        iteration = 0
        while changed and iteration < max_iterations:
            changed1 = step(padded, 1)
            changed2 = step(padded, 2)
            changed = changed1 or changed2
            iteration += 1
        
        # Remove padding
        return padded[1:-1, 1:-1]
    
    def _build_graph(self, skeleton: np.ndarray):
        """
        Build a graph representation of the skeleton.
        
        Returns:
            nodes: Dict mapping node_id to list of connected edge_ids
            edges: Dict mapping edge_id to (node1_id, node2_id, pixel_path)
            node_positions: Dict mapping node_id to (row, col) position
        """
        rows, cols = skeleton.shape
        
        def count_neighbors(r, c):
            """Count 8-connected neighbors that are skeleton pixels."""
            count = 0
            neighbors = []
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and skeleton[nr, nc]:
                        count += 1
                        neighbors.append((nr, nc))
            return count, neighbors
        
        # Identify junction and endpoint pixels
        node_pixels = {}  # (r, c) -> node_id
        node_positions = {}  # node_id -> (r, c)
        node_id_counter = 0
        
        for r in range(rows):
            for c in range(cols):
                if skeleton[r, c]:
                    n_count, _ = count_neighbors(r, c)
                    if n_count != 2:  # Endpoint or junction
                        node_pixels[(r, c)] = node_id_counter
                        node_positions[node_id_counter] = (r, c)
                        node_id_counter += 1
        
        # Trace edges between nodes
        nodes = defaultdict(list)  # node_id -> [edge_ids]
        edges = {}  # edge_id -> (node1_id, node2_id, pixel_path)
        edge_id_counter = 0
        visited_starts = set()
        
        def trace_path(start_r, start_c, prev_r, prev_c):
            """Trace a path from a node pixel until reaching another node."""
            path = [(start_r, start_c)]
            curr_r, curr_c = start_r, start_c
            prev = (prev_r, prev_c)
            
            max_path_len = rows * cols
            
            while len(path) < max_path_len:
                # Check if current pixel is a node
                if (curr_r, curr_c) in node_pixels and len(path) > 1:
                    return node_pixels[(curr_r, curr_c)], path
                
                # Find next pixel
                _, neighbors = count_neighbors(curr_r, curr_c)
                next_pixel = None
                
                for nr, nc in neighbors:
                    if (nr, nc) != prev:
                        next_pixel = (nr, nc)
                        break
                
                if next_pixel is None:
                    if (curr_r, curr_c) in node_pixels:
                        return node_pixels[(curr_r, curr_c)], path
                    return None, path
                
                prev = (curr_r, curr_c)
                curr_r, curr_c = next_pixel
                path.append((curr_r, curr_c))
            
            return None, path
        
        # Trace all edges starting from each node
        for (r, c), node_id in node_pixels.items():
            _, neighbors = count_neighbors(r, c)
            
            for nr, nc in neighbors:
                # Create unique identifier for this edge start
                start_key = ((r, c), (nr, nc))
                reverse_key = ((nr, nc), (r, c))
                
                if start_key in visited_starts or reverse_key in visited_starts:
                    continue
                
                visited_starts.add(start_key)
                
                # Trace this path
                end_node_id, path = trace_path(nr, nc, r, c)
                
                if end_node_id is not None:
                    full_path = [(r, c)] + path
                    edges[edge_id_counter] = (node_id, end_node_id, full_path)
                    nodes[node_id].append(edge_id_counter)
                    nodes[end_node_id].append(edge_id_counter)
                    
                    # Mark reverse direction as visited
                    end_r, end_c = node_positions[end_node_id]
                    if len(path) > 0:
                        last_before_end = path[-2] if len(path) > 1 else (r, c)
                        visited_starts.add(((end_r, end_c), last_before_end))
                    
                    edge_id_counter += 1
        
        return dict(nodes), edges, node_positions
    
    def _calculate_edge_angle(self, edge_id, node_id, edges, node_positions):
        """Calculate the angle of an edge as it leaves a node."""
        node1_id, node2_id, path = edges[edge_id]
        node_pos = node_positions[node_id]
        
        # Get a point along the edge away from the junction
        sample_distance = min(5, len(path) - 1)
        
        if node_id == node1_id:
            sample_pos = path[sample_distance]
        else:
            sample_pos = path[-(sample_distance + 1)]
        
        dy = sample_pos[0] - node_pos[0]
        dx = sample_pos[1] - node_pos[1]
        
        return math.atan2(dy, dx)
    
    def _optimal_pairing(self, angles):
        """
        Find the optimal pairing of 4 edges that minimizes deviation from collinearity.
        
        Args:
            angles: List of 4 (edge_id, angle) tuples
            
        Returns:
            List of two pairs: [(edge1, edge2), (edge3, edge4)]
        """
        if len(angles) != 4:
            return None
        
        edge_ids = [a[0] for a in angles]
        angle_vals = {a[0]: a[1] for a in angles}
        
        def angle_diff(a1, a2):
            """Calculate how close two angles are to being opposite (180° apart)."""
            diff = abs(a1 - a2)
            diff = abs(diff - math.pi)
            return diff
        
        best_pairing = None
        best_score = float('inf')
        
        # All ways to pair 4 items into 2 pairs
        pairings = [
            [(0, 1), (2, 3)],
            [(0, 2), (1, 3)],
            [(0, 3), (1, 2)],
        ]
        
        for pairing in pairings:
            score = 0
            for i, j in pairing:
                e1, e2 = edge_ids[i], edge_ids[j]
                score += angle_diff(angle_vals[e1], angle_vals[e2])
            
            if score < best_score:
                best_score = score
                best_pairing = [(edge_ids[p[0]], edge_ids[p[1]]) for p in pairing]
        
        return best_pairing
    
    def _extract_complete_lines(self, nodes, edges, node_positions):
        """
        Extract individual lines by applying junction rules and tracing.
        
        Returns:
            lines: List of lines, where each line is a list of edge_ids
            line_pixels: List of pixel sets, one for each line
        """
        # Create pairing information for 4-way junctions
        junction_pairings = {}
        
        for node_id, edge_list in nodes.items():
            if len(edge_list) == 4:
                angles = []
                for edge_id in edge_list:
                    angle = self._calculate_edge_angle(edge_id, node_id, edges, node_positions)
                    angles.append((edge_id, angle))
                
                pairing = self._optimal_pairing(angles)
                
                if pairing:
                    junction_pairings[node_id] = {}
                    for e1, e2 in pairing:
                        junction_pairings[node_id][e1] = e2
                        junction_pairings[node_id][e2] = e1
        
        # Trace lines
        lines = []
        line_pixels = []
        used_edges = set()
        
        def get_other_node(edge_id, current_node):
            n1, n2, _ = edges[edge_id]
            return n2 if current_node == n1 else n1
        
        def get_edge_pixels(edge_id):
            """Get all pixels belonging to an edge."""
            _, _, path = edges[edge_id]
            return set(path)
        
        def trace_line(start_edge_id):
            line_edges = [start_edge_id]
            used_edges.add(start_edge_id)
            pixels = get_edge_pixels(start_edge_id)
            
            for direction in [0, 1]:
                n1, n2, _ = edges[start_edge_id]
                current_node = n1 if direction == 0 else n2
                current_edge = start_edge_id
                
                while True:
                    if current_node in junction_pairings:
                        pairing = junction_pairings[current_node]
                        if current_edge in pairing:
                            next_edge = pairing[current_edge]
                            if next_edge not in used_edges:
                                used_edges.add(next_edge)
                                if direction == 0:
                                    line_edges.insert(0, next_edge)
                                else:
                                    line_edges.append(next_edge)
                                pixels.update(get_edge_pixels(next_edge))
                                current_edge = next_edge
                                current_node = get_other_node(next_edge, current_node)
                                continue
                    break
            
            return line_edges, pixels
        
        for edge_id in edges:
            if edge_id not in used_edges:
                line, pixels = trace_line(edge_id)
                lines.append(line)
                line_pixels.append(pixels)
        
        return lines, line_pixels
    
    def _create_colored_image(self, height, width, line_pixels_list, 
                              thickness, binary):
        """
        Create an RGB image with each line colored differently.
        
        Args:
            height: Image height
            width: Image width
            line_pixels_list: List of pixel sets, one per line
            thickness: Line thickness for visualization
            binary: Original binary image for reference
            
        Returns:
            Colored image as numpy array (H, W, 3) with values 0-1
        """
        # Start with white background
        colored = np.ones((height, width, 3), dtype=np.float32)
        
        # Create a mask for dilating skeleton pixels to desired thickness
        def dilate_pixels(pixels, radius):
            """Expand pixel set by given radius."""
            if radius <= 1:
                return pixels
            
            dilated = set()
            for r, c in pixels:
                for dr in range(-radius + 1, radius):
                    for dc in range(-radius + 1, radius):
                        if dr * dr + dc * dc < radius * radius:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < height and 0 <= nc < width:
                                dilated.add((nr, nc))
            return dilated
        
        # Color each line
        for i, pixels in enumerate(line_pixels_list):
            color_idx = i % len(self.LINE_COLORS)
            color = self.LINE_COLORS[color_idx]
            color_normalized = (color[0] / 255.0, color[1] / 255.0, color[2] / 255.0)
            
            # Dilate pixels for thickness
            thick_pixels = dilate_pixels(pixels, thickness)
            
            for r, c in thick_pixels:
                colored[r, c, 0] = color_normalized[0]
                colored[r, c, 1] = color_normalized[1]
                colored[r, c, 2] = color_normalized[2]
        
        return colored
    
class DashedToSolidLine:
    """
    Converts dashed lines in an image to continuous solid lines.
    Supports horizontal, vertical, and diagonal lines.
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "gap_size": ("INT", {
                    "default": 15,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "display": "number"
                }),
                "method": (["morphological", "hough_lines", "hybrid"],),
                "line_thickness": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "display": "number"
                }),
                "threshold": ("INT", {
                    "default": 127,
                    "min": 0,
                    "max": 255,
                    "step": 1,
                    "display": "number"
                }),
                "invert_input": ("BOOLEAN", {"default": False}),
                "skeletonize_output": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "convert_dashed_to_solid"
    CATEGORY = "image/processing"

    def create_diagonal_kernel(self, size, angle):
        """
        Create a diagonal line kernel at specified angle.
        angle: 45 or 135 degrees
        """
        kernel = np.zeros((size, size), dtype=np.uint8)
        
        if angle == 45:
            # Bottom-left to top-right diagonal
            for i in range(size):
                kernel[size - 1 - i, i] = 1
        elif angle == 135:
            # Top-left to bottom-right diagonal
            for i in range(size):
                kernel[i, i] = 1
        
        return kernel

    def morphological_method(self, binary, gap_size, kernel_extra=5):
        """
        Use directional morphological closing to bridge gaps.
        """
        kernel_length = gap_size + kernel_extra
        
        # Ensure odd size for diagonal kernels
        diag_size = kernel_length if kernel_length % 2 == 1 else kernel_length + 1
        
        # Create directional kernels
        kernels = {
            'horizontal': cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1)),
            'vertical': cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length)),
            'diagonal_45': self.create_diagonal_kernel(diag_size, 45),
            'diagonal_135': self.create_diagonal_kernel(diag_size, 135),
        }
        
        # Apply closing with each kernel
        results = []
        for name, kernel in kernels.items():
            closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            results.append(closed)
        
        # Combine all results
        combined = results[0]
        for r in results[1:]:
            combined = cv2.bitwise_or(combined, r)
        
        return combined

    def hough_lines_method(self, binary, gap_size):
        """
        Use Probabilistic Hough Transform to detect and redraw lines.
        Naturally handles all angles.
        """
        # Detect line segments
        lines = cv2.HoughLinesP(
            binary,
            rho=1,
            theta=np.pi / 180,
            threshold=15,
            minLineLength=3,
            maxLineGap=gap_size + 10
        )
        
        # Create output image
        result = np.zeros_like(binary)
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(result, (x1, y1), (x2, y2), 255, 1)
        
        return result

    def hybrid_method(self, binary, gap_size):
        """
        Combine morphological and Hough methods for best results.
        """
        # First pass: morphological to connect obvious gaps
        morph_result = self.morphological_method(binary, gap_size, kernel_extra=3)
        
        # Second pass: Hough to clean up and ensure straight lines
        hough_result = self.hough_lines_method(morph_result, gap_size // 2)
        
        # Combine: use Hough result but fill in any gaps with morphological
        combined = cv2.bitwise_or(hough_result, morph_result)
        
        return combined

    def skeletonize(self, binary):
        """
        Reduce lines to 1-pixel thickness using Zhang-Suen thinning.
        """
        try:
            from skimage.morphology import skeletonize as sk_skeletonize
            skeleton = sk_skeletonize(binary // 255).astype(np.uint8) * 255
            return skeleton
        except ImportError:
            # Fallback: use OpenCV thinning
            return cv2.ximgproc.thinning(binary) if hasattr(cv2, 'ximgproc') else binary

    def convert_dashed_to_solid(self, image, gap_size, method, line_thickness, 
                                 threshold, invert_input, skeletonize_output):
        """
        Main processing function.
        """
        # Convert from ComfyUI tensor format (B, H, W, C) to numpy
        batch_size = image.shape[0]
        results = []
        
        for b in range(batch_size):
            # Extract single image and convert to numpy uint8
            img = image[b].cpu().numpy()
            img = (img * 255).astype(np.uint8)
            
            # Convert to grayscale if needed
            if len(img.shape) == 3 and img.shape[2] >= 3:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                gray = img
            
            # Binarize
            if invert_input:
                _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
            else:
                _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
            
            # Apply selected method
            if method == "morphological":
                processed = self.morphological_method(binary, gap_size)
            elif method == "hough_lines":
                processed = self.hough_lines_method(binary, gap_size)
            else:  # hybrid
                processed = self.hybrid_method(binary, gap_size)
            
            # Skeletonize if requested
            if skeletonize_output:
                processed = self.skeletonize(processed)
            
            # Apply line thickness
            if line_thickness > 1:
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, 
                    (line_thickness, line_thickness)
                )
                processed = cv2.dilate(processed, kernel, iterations=1)
            
            # Invert to get dark lines on white background
            processed = cv2.bitwise_not(processed)
            
            # Convert back to RGB
            result_rgb = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
            
            # Normalize to 0-1 float
            result_float = result_rgb.astype(np.float32) / 255.0
            
            results.append(result_float)
        
        # Stack back to batch tensor
        result_batch = np.stack(results, axis=0)
        result_tensor = torch.from_numpy(result_batch)
        
        return (result_tensor,)