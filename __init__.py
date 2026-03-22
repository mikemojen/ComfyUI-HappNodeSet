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
from .linecounter import *
from .numstrcleanup import *
from .laserpath import *

NODE_CLASS_MAPPINGS = {
    "RasterToUniformSVG": RasterToSVGConverter,
    "SVGDimensions" : SVGDimensionNode,
    "SVGPathLength" : SVGPathLengthCalculator,
    "SVGPathLengthDetailed" : SVGPathLengthDetailed,
    "ColorQuantizer" : ImageColorQuantizer,
    "ExtractBlack": ExtractBlackColor,
    "ExtractRed" : ExtractRedColor,
    "ExtractBlackAdvanced" : ExtractBlackColorAdvanced,
    "NonWhiteToBlack": NonWhiteToBlack,
    "DashedToSolidLine": DashedToSolidLine,
    "LineDetector": LineDetectorNode,
    "AutoCrop": AutoCropNode,
    "RoundUpNode": RoundUpNode,
    "RoundDownNode": RoundDownNode,
    "CSVValueExtractor": CSVValueExtractor,
    "NumberStringCleanup": NumberStringCleanup,
    "LaserPathTracer": LaserPathTracerNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RasterToUniformSVG" : "Raster To Uniform SVG",  
    "SVGDimensions" : "SVG Dimensions",
    "SVGPathLength" : "SVG PathLength",
    "SVGPathLengthDetailed" : "SVG Detailed PathLength",
    "ColorQuantizer" : "Color Quantizer",
    "ExtractBlack": "Extract Black",
    "ExtractRed" : "Extract Red",
    "ExtractBlackAdvanced" : "Extract Black Advanced",
    "NonWhiteToBlack": "Non-White to Black",
    "DashedToSolidLine": "Dashed to Solid Line",
    "LineDetector": "Line Detector",
    "AutoCrop": "Auto Crop",
    "RoundUpNode": "Round Up",
    "RoundDownNode": "Round Down",
    "CSVValueExtractor": "CSV Value Extractor",
    "NumberStringCleanup": "Number String Cleanup",
    "LaserPathTracer": "Laser Cutter Path Tracer",
}