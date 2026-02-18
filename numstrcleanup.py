"""
SVG Path Length Calculator Node for ComfyUI

Calculates the total length of all paths in an SVG file,
ignoring line thickness (stroke-width).
"""

import re


class NumberStringCleanup:
    """
    Cleans up a raw string containing a number and converts it
    into a two-decimal-place float-compatible string.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "raw_string": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("cleaned_number",)
    FUNCTION = "clean_number"
    CATEGORY = "utils/string"

    def clean_number(self, raw_string: str):
        s = raw_string.strip()

        # Remove everything except digits, commas, dots, and minus sign
        s = re.sub(r"[^\d.,-]", "", s)

        # Replace comma with dot (e.g. "52,25" -> "52.25")
        s = s.replace(",", ".")

        # If multiple dots exist, keep only the last one as decimal separator
        if s.count(".") > 1:
            parts = s.split(".")
            s = "".join(parts[:-1]) + "." + parts[-1]

        # Handle edge cases
        if s in ("", ".", "-", "-."):
            return ("0.00",)

        # Handle leading minus
        negative = s.startswith("-")
        if negative:
            s = s[1:]

        # Convert to float and format to 2 decimal places
        try:
            value = float(s)
        except ValueError:
            value = 0.0

        if negative:
            value = -value

        result = f"{value:.2f}"
        return (result,)