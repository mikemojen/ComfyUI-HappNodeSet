"""
comfyUI nodes for Happrecision
"""

from .rastersvg import *
from .svgpathlength import *
from .svgdimension import *
from .colorquantize import *
from .blackextract import *
from .autocrop import *
from .numfunction import *

NODE_CLASS_MAPPINGS = {
    "RasterToUniformSVG": RasterToSVGConverter,
    "SVGDimensions" : SVGDimensionNode,
    "SVGPathLength" : SVGPathLengthCalculator,
    "SVGPathLengthDetailed" : SVGPathLengthDetailed,
    "ColorQuantizer" : ImageColorQuantizer,
    "ExtractBlack": ExtractBlackColor,
    "ExtractBlackAdvanced" : ExtractBlackColorAdvanced,
    "AutoCrop": AutoCropNode,
    "RoundUpNode": RoundUpNode,
    "RoundDownNode": RoundDownNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RasterToUniformSVG" : "Raster To Uniform SVG",  
    "SVGDimensions" : "SVG Dimensions",
    "SVGPathLength" : "SVG PathLength",
    "SVGPathLengthDetailed" : "SVG Detailed PathLength",
    "ColorQuantizer" : "Color Quantizer",
    "ExtractBlack": "Extract Black",
    "ExtractBlackAdvanced" : "Extract Black Advanced",
    "AutoCrop": "Auto Crop",
    "RoundUpNode": "Round Up",
    "RoundDownNode": "Round Down"
}