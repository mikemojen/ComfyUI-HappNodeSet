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