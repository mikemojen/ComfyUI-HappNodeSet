"""
SVG Path Length Calculator Node for ComfyUI

Calculates the total length of all paths in an SVG file,
ignoring line thickness (stroke-width).
"""

import re
import math


def parse_path_d(d: str) -> list:
    """Parse SVG path 'd' attribute into a list of commands and coordinates."""
    tokens = re.findall(r'[MmZzLlHhVvCcSsQqTtAa]|[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?', d)
    
    commands = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token.isalpha():
            cmd = token
            i += 1
            params = []
            while i < len(tokens) and not tokens[i].isalpha():
                params.append(float(tokens[i]))
                i += 1
            commands.append((cmd, params))
        else:
            i += 1
    
    return commands


def calculate_bezier_length(points: list, segments: int = 100) -> float:
    """Calculate approximate length of a Bezier curve using line segments."""
    length = 0.0
    prev_point = None
    
    for t in range(segments + 1):
        t_val = t / segments
        point = bezier_point(points, t_val)
        if prev_point is not None:
            length += math.dist(prev_point, point)
        prev_point = point
    
    return length


def bezier_point(points: list, t: float) -> tuple:
    """Calculate a point on a Bezier curve at parameter t."""
    n = len(points) - 1
    x = 0.0
    y = 0.0
    
    for i, (px, py) in enumerate(points):
        coef = math.comb(n, i) * (1 - t) ** (n - i) * t ** i
        x += coef * px
        y += coef * py
    
    return (x, y)


def calculate_arc_length(rx: float, ry: float, x_axis_rotation: float,
                         large_arc: bool, sweep: bool,
                         start: tuple, end: tuple) -> float:
    """Calculate approximate length of an elliptical arc."""
    if rx == 0 or ry == 0:
        return math.dist(start, end)
    
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    chord = math.sqrt(dx * dx + dy * dy)
    
    avg_r = (abs(rx) + abs(ry)) / 2
    if chord > 2 * avg_r:
        return chord
    
    angle = 2 * math.asin(min(chord / (2 * avg_r), 1.0))
    if large_arc:
        angle = 2 * math.pi - angle
    
    return abs(angle * avg_r)


def calculate_path_length(d: str) -> float:
    """Calculate the total length of an SVG path."""
    if not d:
        return 0.0
    
    commands = parse_path_d(d)
    
    total_length = 0.0
    current_x, current_y = 0.0, 0.0
    start_x, start_y = 0.0, 0.0
    last_control = None
    last_command = None
    
    for cmd, params in commands:
        is_relative = cmd.islower()
        cmd_upper = cmd.upper()
        
        if cmd_upper == 'M':
            i = 0
            while i < len(params):
                x, y = params[i], params[i + 1]
                if is_relative:
                    x += current_x
                    y += current_y
                
                if i == 0:
                    start_x, start_y = x, y
                else:
                    total_length += math.dist((current_x, current_y), (x, y))
                
                current_x, current_y = x, y
                i += 2
            last_control = None
            
        elif cmd_upper == 'L':
            i = 0
            while i < len(params):
                x, y = params[i], params[i + 1]
                if is_relative:
                    x += current_x
                    y += current_y
                total_length += math.dist((current_x, current_y), (x, y))
                current_x, current_y = x, y
                i += 2
            last_control = None
            
        elif cmd_upper == 'H':
            for x in params:
                if is_relative:
                    x += current_x
                total_length += abs(x - current_x)
                current_x = x
            last_control = None
            
        elif cmd_upper == 'V':
            for y in params:
                if is_relative:
                    y += current_y
                total_length += abs(y - current_y)
                current_y = y
            last_control = None
            
        elif cmd_upper == 'Z':
            total_length += math.dist((current_x, current_y), (start_x, start_y))
            current_x, current_y = start_x, start_y
            last_control = None
            
        elif cmd_upper == 'C':
            i = 0
            while i + 5 < len(params) or i + 5 == len(params):
                if i + 5 >= len(params) and i + 5 != len(params):
                    break
                x1, y1 = params[i], params[i + 1]
                x2, y2 = params[i + 2], params[i + 3]
                x, y = params[i + 4], params[i + 5]
                
                if is_relative:
                    x1 += current_x
                    y1 += current_y
                    x2 += current_x
                    y2 += current_y
                    x += current_x
                    y += current_y
                
                points = [(current_x, current_y), (x1, y1), (x2, y2), (x, y)]
                total_length += calculate_bezier_length(points)
                last_control = (x2, y2)
                current_x, current_y = x, y
                i += 6
            
        elif cmd_upper == 'S':
            i = 0
            while i + 3 < len(params) or i + 3 == len(params):
                if i + 3 >= len(params) and i + 3 != len(params):
                    break
                x2, y2 = params[i], params[i + 1]
                x, y = params[i + 2], params[i + 3]
                
                if is_relative:
                    x2 += current_x
                    y2 += current_y
                    x += current_x
                    y += current_y
                
                if last_command in ('C', 'c', 'S', 's') and last_control:
                    x1 = 2 * current_x - last_control[0]
                    y1 = 2 * current_y - last_control[1]
                else:
                    x1, y1 = current_x, current_y
                
                points = [(current_x, current_y), (x1, y1), (x2, y2), (x, y)]
                total_length += calculate_bezier_length(points)
                last_control = (x2, y2)
                current_x, current_y = x, y
                i += 4
            
        elif cmd_upper == 'Q':
            i = 0
            while i + 3 < len(params) or i + 3 == len(params):
                if i + 3 >= len(params) and i + 3 != len(params):
                    break
                x1, y1 = params[i], params[i + 1]
                x, y = params[i + 2], params[i + 3]
                
                if is_relative:
                    x1 += current_x
                    y1 += current_y
                    x += current_x
                    y += current_y
                
                points = [(current_x, current_y), (x1, y1), (x, y)]
                total_length += calculate_bezier_length(points)
                last_control = (x1, y1)
                current_x, current_y = x, y
                i += 4
            
        elif cmd_upper == 'T':
            i = 0
            while i + 1 < len(params) or i + 1 == len(params):
                if i + 1 >= len(params) and i + 1 != len(params):
                    break
                x, y = params[i], params[i + 1]
                
                if is_relative:
                    x += current_x
                    y += current_y
                
                if last_command in ('Q', 'q', 'T', 't') and last_control:
                    x1 = 2 * current_x - last_control[0]
                    y1 = 2 * current_y - last_control[1]
                else:
                    x1, y1 = current_x, current_y
                
                points = [(current_x, current_y), (x1, y1), (x, y)]
                total_length += calculate_bezier_length(points)
                last_control = (x1, y1)
                current_x, current_y = x, y
                i += 2
            
        elif cmd_upper == 'A':
            i = 0
            while i + 6 < len(params) or i + 6 == len(params):
                if i + 6 >= len(params) and i + 6 != len(params):
                    break
                rx = params[i]
                ry = params[i + 1]
                x_rotation = params[i + 2]
                large_arc = bool(params[i + 3])
                sweep = bool(params[i + 4])
                x, y = params[i + 5], params[i + 6]
                
                if is_relative:
                    x += current_x
                    y += current_y
                
                arc_len = calculate_arc_length(
                    rx, ry, x_rotation, large_arc, sweep,
                    (current_x, current_y), (x, y)
                )
                total_length += arc_len
                current_x, current_y = x, y
                i += 7
            last_control = None
        
        last_command = cmd
    
    return total_length


def get_transform_scale(transform: str) -> tuple:
    """Extract scale factors from a transform attribute."""
    if not transform:
        return (1.0, 1.0)
    
    scale_x, scale_y = 1.0, 1.0
    
    scale_match = re.search(r'scale\s*\(\s*([-\d.]+)(?:\s*[,\s]\s*([-\d.]+))?\s*\)', transform)
    if scale_match:
        scale_x = float(scale_match.group(1))
        scale_y = float(scale_match.group(2)) if scale_match.group(2) else scale_x
    
    matrix_match = re.search(r'matrix\s*\(\s*([-\d.]+)\s*[,\s]\s*([-\d.]+)\s*[,\s]\s*([-\d.]+)\s*[,\s]\s*([-\d.]+)', transform)
    if matrix_match:
        a = float(matrix_match.group(1))
        b = float(matrix_match.group(2))
        c = float(matrix_match.group(3))
        d = float(matrix_match.group(4))
        scale_x = math.sqrt(a * a + b * b)
        scale_y = math.sqrt(c * c + d * d)
    
    return (scale_x, scale_y)


def calculate_shape_perimeter(element, ns: dict) -> float:
    """Calculate the perimeter of basic SVG shapes."""
    tag = element.tag.split('}')[-1] if '}' in element.tag else element.tag
    
    if tag == 'rect':
        width = float(element.get('width', 0))
        height = float(element.get('height', 0))
        rx = float(element.get('rx', 0))
        ry = float(element.get('ry', rx))
        
        if rx == 0 and ry == 0:
            return 2 * (width + height)
        else:
            straight = 2 * (width - 2 * rx) + 2 * (height - 2 * ry)
            h = ((rx - ry) ** 2) / ((rx + ry) ** 2)
            corner_perimeter = math.pi * (rx + ry) * (1 + 3 * h / (10 + math.sqrt(4 - 3 * h)))
            return straight + corner_perimeter
    
    elif tag == 'circle':
        r = float(element.get('r', 0))
        return 2 * math.pi * r
    
    elif tag == 'ellipse':
        rx = float(element.get('rx', 0))
        ry = float(element.get('ry', 0))
        h = ((rx - ry) ** 2) / ((rx + ry) ** 2)
        return math.pi * (rx + ry) * (1 + 3 * h / (10 + math.sqrt(4 - 3 * h)))
    
    elif tag == 'line':
        x1 = float(element.get('x1', 0))
        y1 = float(element.get('y1', 0))
        x2 = float(element.get('x2', 0))
        y2 = float(element.get('y2', 0))
        return math.dist((x1, y1), (x2, y2))
    
    elif tag == 'polyline':
        points_str = element.get('points', '')
        points = parse_points(points_str)
        return calculate_polyline_length(points, closed=False)
    
    elif tag == 'polygon':
        points_str = element.get('points', '')
        points = parse_points(points_str)
        return calculate_polyline_length(points, closed=True)
    
    return 0.0


def parse_points(points_str: str) -> list:
    """Parse SVG points attribute into a list of (x, y) tuples."""
    numbers = re.findall(r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?', points_str)
    points = []
    for i in range(0, len(numbers) - 1, 2):
        points.append((float(numbers[i]), float(numbers[i + 1])))
    return points


def calculate_polyline_length(points: list, closed: bool = False) -> float:
    """Calculate the length of a polyline or polygon."""
    if len(points) < 2:
        return 0.0
    
    length = 0.0
    for i in range(len(points) - 1):
        length += math.dist(points[i], points[i + 1])
    
    if closed and len(points) > 2:
        length += math.dist(points[-1], points[0])
    
    return length


def calculate_svg_total_length(svg_content: str) -> float:
    """Calculate the total length of all paths in an SVG."""
    from xml.etree import ElementTree as ET
    
    try:
        root = ET.fromstring(svg_content)
    except ET.ParseError as e:
        raise ValueError(f"Invalid SVG: {e}")
    
    ns = {}
    if root.tag.startswith('{'):
        ns_uri = root.tag[1:root.tag.index('}')]
        ns['svg'] = ns_uri
    
    total_length = 0.0
    
    def process_element(element, inherited_scale=(1.0, 1.0)):
        nonlocal total_length
        
        transform = element.get('transform', '')
        scale = get_transform_scale(transform)
        current_scale = (inherited_scale[0] * scale[0], inherited_scale[1] * scale[1])
        avg_scale = (current_scale[0] + current_scale[1]) / 2
        
        tag = element.tag.split('}')[-1] if '}' in element.tag else element.tag
        
        if tag == 'path':
            d = element.get('d', '')
            if d:
                path_length = calculate_path_length(d)
                total_length += path_length * avg_scale
        
        elif tag in ('rect', 'circle', 'ellipse', 'line', 'polyline', 'polygon'):
            shape_length = calculate_shape_perimeter(element, ns)
            total_length += shape_length * avg_scale
        
        for child in element:
            process_element(child, current_scale)
    
    process_element(root)
    
    return total_length


class SVGPathLengthCalculator:
    """
    ComfyUI node that calculates the total length of all paths in an SVG file.
    Supports path commands, basic shapes (rect, circle, ellipse, line, polyline, polygon),
    and handles transforms.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "svg_content": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Paste SVG content here or connect from another node"
                }),
            },
            "optional": {
                "decimal_places": ("INT", {
                    "default": 4,
                    "min": 0,
                    "max": 10,
                    "step": 1,
                    "display": "number"
                }),
            }
        }
    
    RETURN_TYPES = ("FLOAT", "STRING",)
    RETURN_NAMES = ("total_length", "formatted_result",)
    FUNCTION = "calculate_length"
    CATEGORY = "utils/svg"
    OUTPUT_NODE = True
    
    def calculate_length(self, svg_content: str, decimal_places: int = 4):
        """
        Calculate the total path length of an SVG.
        
        Args:
            svg_content: The SVG file content as a string
            decimal_places: Number of decimal places for the result
            
        Returns:
            Tuple of (total_length as float, formatted result string)
        """
        if not svg_content or not svg_content.strip():
            return (0.0, "Error: No SVG content provided")
        
        try:
            total_length = calculate_svg_total_length(svg_content)
            formatted = f"Total path length: {total_length:.{decimal_places}f} units"
            return (round(total_length, decimal_places), formatted)
        except ValueError as e:
            return (0.0, f"Error: {str(e)}")
        except Exception as e:
            return (0.0, f"Error processing SVG: {str(e)}")


class SVGPathLengthDetailed:
    """
    ComfyUI node that calculates path lengths with detailed breakdown per element.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "svg_content": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Paste SVG content here"
                }),
            },
            "optional": {
                "decimal_places": ("INT", {
                    "default": 4,
                    "min": 0,
                    "max": 10,
                    "step": 1,
                    "display": "number"
                }),
            }
        }
    
    RETURN_TYPES = ("FLOAT", "STRING", "INT",)
    RETURN_NAMES = ("total_length", "detailed_report", "element_count",)
    FUNCTION = "calculate_detailed"
    CATEGORY = "utils/svg"
    OUTPUT_NODE = True
    
    def calculate_detailed(self, svg_content: str, decimal_places: int = 4):
        """
        Calculate path lengths with a detailed breakdown.
        
        Returns:
            Tuple of (total_length, detailed report string, element count)
        """
        from xml.etree import ElementTree as ET
        
        if not svg_content or not svg_content.strip():
            return (0.0, "Error: No SVG content provided", 0)
        
        try:
            root = ET.fromstring(svg_content)
        except ET.ParseError as e:
            return (0.0, f"Error: Invalid SVG - {e}", 0)
        
        elements_info = []
        total_length = 0.0
        
        def process_element(element, inherited_scale=(1.0, 1.0), depth=0):
            nonlocal total_length
            
            transform = element.get('transform', '')
            scale = get_transform_scale(transform)
            current_scale = (inherited_scale[0] * scale[0], inherited_scale[1] * scale[1])
            avg_scale = (current_scale[0] + current_scale[1]) / 2
            
            tag = element.tag.split('}')[-1] if '}' in element.tag else element.tag
            element_length = 0.0
            
            if tag == 'path':
                d = element.get('d', '')
                if d:
                    element_length = calculate_path_length(d) * avg_scale
                    element_id = element.get('id', 'unnamed')
                    elements_info.append((tag, element_id, element_length))
                    total_length += element_length
            
            elif tag in ('rect', 'circle', 'ellipse', 'line', 'polyline', 'polygon'):
                element_length = calculate_shape_perimeter(element, {}) * avg_scale
                element_id = element.get('id', 'unnamed')
                elements_info.append((tag, element_id, element_length))
                total_length += element_length
            
            for child in element:
                process_element(child, current_scale, depth + 1)
        
        try:
            process_element(root)
        except Exception as e:
            return (0.0, f"Error processing SVG: {str(e)}", 0)
        
        # Build detailed report
        report_lines = ["SVG Path Length Analysis", "=" * 40]
        
        for tag, elem_id, length in elements_info:
            report_lines.append(f"  {tag} (id='{elem_id}'): {length:.{decimal_places}f} units")
        
        report_lines.append("=" * 40)
        report_lines.append(f"Total elements: {len(elements_info)}")
        report_lines.append(f"Total length: {total_length:.{decimal_places}f} units")
        
        detailed_report = "\n".join(report_lines)
        
        return (round(total_length, decimal_places), detailed_report, len(elements_info))