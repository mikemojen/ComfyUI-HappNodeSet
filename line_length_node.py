import numpy as np
import torch
from PIL import Image
import cv2
from skimage.morphology import skeletonize


class LineLengthCalculator:
    """
    Calculates the total length of black lines in a diagram image.
    
    Works on images with white/transparent backgrounds and black lines.
    Uses skeletonization to reduce lines to 1px width, then measures
    length accounting for diagonal (√2) vs orthogonal (1.0) pixel steps.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "threshold": ("INT", {
                    "default": 128,
                    "min": 1,
                    "max": 254,
                    "step": 1,
                    "tooltip": "Pixel intensity threshold (0-255). Pixels darker than this are treated as line pixels."
                }),
                "invert": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable if lines are white on dark background instead of black on light."
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "FLOAT", "INT", "STRING")
    RETURN_NAMES = ("skeleton_overlay", "total_length_px", "skeleton_pixel_count", "report")
    FUNCTION = "calculate_line_length"
    CATEGORY = "diagram-analysis"
    DESCRIPTION = "Calculates total length of black lines in a diagram image (in pixels). Returns a skeleton visualization, the measured length, raw skeleton pixel count, and a text report."

    def calculate_line_length(self, image: torch.Tensor, threshold: int = 128, invert: bool = False):
        # ---- 1. Convert ComfyUI image tensor → numpy uint8 ----
        # ComfyUI images: (B, H, W, C) float32 [0,1] RGB
        img_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
        h, w = img_np.shape[:2]

        # ---- 2. Handle alpha channel (transparent → white) ----
        if img_np.shape[2] == 4:
            alpha = img_np[:, :, 3].astype(np.float32) / 255.0
            rgb = img_np[:, :, :3].astype(np.float32)
            white_bg = np.ones_like(rgb) * 255.0
            # Composite: foreground * alpha + white * (1-alpha)
            composited = rgb * alpha[:, :, np.newaxis] + white_bg * (1.0 - alpha[:, :, np.newaxis])
            img_np = composited.astype(np.uint8)
        else:
            img_np = img_np[:, :, :3]

        # ---- 3. Grayscale + binary threshold ----
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        if invert:
            # White lines on dark bg → lines are bright
            binary = (gray > threshold).astype(np.uint8)
        else:
            # Black lines on light bg → lines are dark
            binary = (gray < threshold).astype(np.uint8)

        # ---- 4. Skeletonize ----
        skeleton = skeletonize(binary > 0)  # returns bool array
        skeleton_uint8 = skeleton.astype(np.uint8)

        # ---- 5. Calculate length using neighbor-aware method ----
        # For each skeleton pixel, count its 8-connected neighbors that are
        # also skeleton pixels. Use Freeman chain code logic:
        #   - orthogonal neighbor (up/down/left/right) contributes 0.5 each side → 1.0 per link
        #   - diagonal neighbor contributes √2/2 each side → √2 per link
        # Total length = sum over all skeleton pixels of half the weighted neighbor count
        # (each link counted from both endpoints, so divide by 2)

        skel = skeleton_uint8
        total_length = 0.0
        skeleton_count = int(np.sum(skel))

        if skeleton_count > 0:
            # Pad to avoid boundary checks
            padded = np.pad(skel, 1, mode='constant', constant_values=0)

            # 8 neighbor offsets: (dy, dx, distance)
            neighbors = [
                (-1, -1, np.sqrt(2)),  # top-left
                (-1,  0, 1.0),         # top
                (-1,  1, np.sqrt(2)),  # top-right
                ( 0, -1, 1.0),         # left
                ( 0,  1, 1.0),         # right
                ( 1, -1, np.sqrt(2)),  # bottom-left
                ( 1,  0, 1.0),         # bottom
                ( 1,  1, np.sqrt(2)),  # bottom-right
            ]

            # Get coordinates of all skeleton pixels (in padded image)
            ys, xs = np.where(padded == 1)

            for dy, dx, dist in neighbors:
                # Check how many skeleton pixels have a neighbor at (dy, dx)
                neighbor_vals = padded[ys + dy, xs + dx]
                total_length += np.sum(neighbor_vals) * dist

            # Each edge counted twice (once from each endpoint)
            total_length /= 2.0

        # ---- 6. Build skeleton overlay visualization ----
        # Red skeleton on top of the original image
        overlay = img_np.copy()
        overlay[skeleton, 0] = 255  # R
        overlay[skeleton, 1] = 0    # G
        overlay[skeleton, 2] = 0    # B

        # Convert back to ComfyUI tensor
        overlay_tensor = torch.from_numpy(overlay.astype(np.float32) / 255.0).unsqueeze(0)

        # ---- 7. Build report string ----
        report = (
            f"=== Line Length Report ===\n"
            f"Image size: {w} x {h} px\n"
            f"Threshold: {threshold}\n"
            f"Black pixels (pre-skeleton): {int(np.sum(binary))}\n"
            f"Skeleton pixels: {skeleton_count}\n"
            f"Total line length: {total_length:.1f} px\n"
            f"========================="
        )

        return (overlay_tensor, round(total_length, 2), skeleton_count, report)
