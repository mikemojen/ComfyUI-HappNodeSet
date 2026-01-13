import math

class RoundUpNode:
    """
    A ComfyUI custom node that rounds up any float to the next integer.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("FLOAT", {
                    "default": 0.0,
                    "min": -1e10,
                    "max": 1e10,
                    "step": 0.01,
                }),
            },
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("integer",)
    FUNCTION = "roundup"
    CATEGORY = "math"

    def roundup(self, value):
        result = math.ceil(value)
        return (result,)
    
class RoundDownNode:
    """
    A ComfyUI custom node that rounds down any float to the previous integer.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("FLOAT", {
                    "default": 0.0,
                    "min": -1e10,
                    "max": 1e10,
                    "step": 0.01,
                }),
            },
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("integer",)
    FUNCTION = "rounddown"
    CATEGORY = "math"

    def rounddown(self, value):
        result = math.floor(value)
        return (result,)
    
class CSVValueExtractor:
    """
    A ComfyUI custom node that extracts a single value from a comma-separated string by index.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "csv_string": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "e.g., 824.00, 823.00, 446.00, 446.00"
                }),
                "index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 1000,
                    "step": 1,
                    "display": "number"
                }),
            },
        }

    RETURN_TYPES = ("FLOAT", "STRING",)
    RETURN_NAMES = ("value_float", "value_string",)
    FUNCTION = "extract_value"
    CATEGORY = "utils"

    def extract_value(self, csv_string, index):
        # Split the string by comma and strip whitespace
        values = [v.strip() for v in csv_string.split(",") if v.strip()]
        
        # Return 0 if empty or index out of range
        if not values or index < 0 or index >= len(values):
            return (0.0, "0",)
        
        value_str = values[index]
        
        # Try to convert to float, return 0 if conversion fails
        try:
            value_float = float(value_str)
        except ValueError:
            return (0.0, "0",)
        
        return (value_float, value_str,)