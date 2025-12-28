"""
ComfyUI Custom Node: SVG Dimension Extractor
Accepts SVG input (as string or file path) and returns width and height dimensions.
"""

import re
import xml.etree.ElementTree as ET
from typing import Tuple, Optional


class SVGDimensionNode:
    """
    A ComfyUI custom node that extracts width and height dimensions from SVG content.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "svg_input": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Paste SVG content here or provide file path"
                }),
            },
            "optional": {
                "default_unit": (["px", "pt", "em", "rem", "cm", "mm", "in"], {
                    "default": "px"
                }),
            }
        }
    
    RETURN_TYPES = ("FLOAT", "FLOAT", "STRING", "STRING")
    RETURN_NAMES = ("width", "height", "width_with_unit", "height_with_unit")
    FUNCTION = "measure_svg"
    CATEGORY = "utils/svg"
    
    # Class constants for unit conversion to pixels
    UNIT_TO_PX = {
        "px": 1.0,
        "pt": 1.333333,  # 1pt = 1.333px (96/72)
        "em": 16.0,      # Assuming 1em = 16px (default browser font size)
        "rem": 16.0,     # Assuming 1rem = 16px
        "cm": 37.795276, # 1cm = 37.795276px (96/2.54)
        "mm": 3.779528,  # 1mm = 3.779528px
        "in": 96.0,      # 1in = 96px
        "%": 1.0,        # Percentage - context dependent, return as-is
    }

    def parse_dimension(self, value: str) -> Tuple[float, str]:
        """
        Parse a dimension string and extract numeric value and unit.
        
        Args:
            value: Dimension string like "100px", "50%", "10em", etc.
            
        Returns:
            Tuple of (numeric_value, unit_string)
        """
        if not value:
            return (0.0, "")
        
        value = value.strip()
        
        # Regular expression to match number (including decimals) and optional unit
        match = re.match(r'^([+-]?[\d.]+)\s*([a-zA-Z%]*)$', value)
        
        if match:
            num_str = match.group(1)
            unit = match.group(2).lower() if match.group(2) else "px"
            try:
                num = float(num_str)
                return (num, unit)
            except ValueError:
                return (0.0, "")
        
        return (0.0, "")

    def parse_viewbox(self, viewbox: str) -> Tuple[float, float, float, float]:
        """
        Parse SVG viewBox attribute.
        
        Args:
            viewbox: viewBox string like "0 0 100 100"
            
        Returns:
            Tuple of (min_x, min_y, width, height)
        """
        if not viewbox:
            return (0.0, 0.0, 0.0, 0.0)
        
        # Split by whitespace or comma
        parts = re.split(r'[\s,]+', viewbox.strip())
        
        if len(parts) >= 4:
            try:
                return (
                    float(parts[0]),
                    float(parts[1]),
                    float(parts[2]),
                    float(parts[3])
                )
            except ValueError:
                pass
        
        return (0.0, 0.0, 0.0, 0.0)

    def convert_to_pixels(self, value: float, unit: str) -> float:
        """
        Convert a dimension value to pixels.
        
        Args:
            value: Numeric value
            unit: Unit string
            
        Returns:
            Value in pixels
        """
        multiplier = self.UNIT_TO_PX.get(unit, 1.0)
        return value * multiplier

    def measure_svg(self, svg_input: str, default_unit: str = "px") -> Tuple[float, float, str, str]:
        """
        Measure the dimensions of an SVG.
        
        Args:
            svg_input: SVG content as string or file path
            default_unit: Default unit to use if none specified
            
        Returns:
            Tuple of (width_px, height_px, width_with_unit, height_with_unit)
        """
        svg_content = svg_input.strip()
        
        # Check if input is a file path
        if not svg_content.startswith('<') and not svg_content.startswith('<?'):
            try:
                with open(svg_content, 'r', encoding='utf-8') as f:
                    svg_content = f.read()
            except (FileNotFoundError, IOError, OSError):
                # Not a valid file path, treat as SVG content
                pass
        
        # Parse the SVG
        try:
            # Handle SVG namespace
            root = ET.fromstring(svg_content)
        except ET.ParseError as e:
            print(f"SVG parsing error: {e}")
            return (0.0, 0.0, "0px", "0px")
        
        # Extract attributes (handle namespace)
        # SVG namespace
        ns = {'svg': 'http://www.w3.org/2000/svg'}
        
        # Get width and height attributes
        width_attr = root.get('width', '')
        height_attr = root.get('height', '')
        viewbox_attr = root.get('viewBox', '') or root.get('viewbox', '')
        
        # Parse viewBox for fallback dimensions
        vb_min_x, vb_min_y, vb_width, vb_height = self.parse_viewbox(viewbox_attr)
        
        # Parse width
        if width_attr:
            width_value, width_unit = self.parse_dimension(width_attr)
            if not width_unit:
                width_unit = default_unit
        elif vb_width > 0:
            # Fall back to viewBox width
            width_value = vb_width
            width_unit = default_unit
        else:
            width_value = 0.0
            width_unit = default_unit
        
        # Parse height
        if height_attr:
            height_value, height_unit = self.parse_dimension(height_attr)
            if not height_unit:
                height_unit = default_unit
        elif vb_height > 0:
            # Fall back to viewBox height
            height_value = vb_height
            height_unit = default_unit
        else:
            height_value = 0.0
            height_unit = default_unit
        
        # Convert to pixels for numeric output
        width_px = self.convert_to_pixels(width_value, width_unit)
        height_px = self.convert_to_pixels(height_value, height_unit)
        
        # Format strings with units
        width_with_unit = f"{width_value}{width_unit}"
        height_with_unit = f"{height_value}{height_unit}"
        
        return (width_px, height_px, width_with_unit, height_with_unit)