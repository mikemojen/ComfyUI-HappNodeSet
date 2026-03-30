"""
ComfyUI Custom Node: Contour Gap Closer

Detects broken contour lines in binary/sketch images and connects
nearby endpoints to produce closed, continuous contours.

Algorithm:
  1. Binarize the input image (contour lines = foreground).
  2. Skeletonize to single-pixel-wide lines.
  3. Detect endpoints (pixels with exactly 1 neighbour in the skeleton).
  4. For every endpoint, find the nearest other endpoint within a
     configurable max-gap distance and draw a connecting line.
  5. Optionally apply a second skeletonize pass so the output stays
     single-pixel-wide.
  6. Optionally apply morphological closing first to heal very small
     micro-gaps before the endpoint search.
"""

import numpy as np
import torch
from scipy import ndimage
from scipy.spatial import KDTree
from skimage.morphology import skeletonize, disk, closing
from skimage.draw import line as sk_line


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _binarize(gray: np.ndarray, threshold: float, invert: bool) -> np.ndarray:
    """Return a bool array where True = contour foreground."""
    if invert:
        return gray < threshold
    return gray > threshold


def _find_endpoints(skel: np.ndarray) -> np.ndarray:
    """
    Find endpoints in a skeletonised binary image.
    An endpoint is a foreground pixel with exactly 1 foreground neighbour
    in its 8-connected neighbourhood.
    Returns an (N, 2) int array of (row, col) coordinates.
    """
    # Count neighbours using convolution
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)
    neighbour_count = ndimage.convolve(skel.astype(np.uint8), kernel,
                                       mode='constant', cval=0)
    endpoints = np.argwhere(skel & (neighbour_count == 1))
    return endpoints


def _direction_at_endpoint(skel: np.ndarray, pt: np.ndarray,
                           look_ahead: int = 6) -> np.ndarray:
    """
    Estimate the tangent direction of the contour at an endpoint by
    walking `look_ahead` pixels along the skeleton from `pt`.
    Returns a unit vector (dr, dc).
    """
    r, c = int(pt[0]), int(pt[1])
    h, w = skel.shape
    visited = set()
    visited.add((r, c))
    path = [(r, c)]

    for _ in range(look_ahead):
        found_next = False
        cr, cc = path[-1]
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w:
                    if skel[nr, nc] and (nr, nc) not in visited:
                        visited.add((nr, nc))
                        path.append((nr, nc))
                        found_next = True
                        break
            if found_next:
                break
        if not found_next:
            break

    if len(path) < 2:
        return np.array([0.0, 0.0])

    # Direction vector from last path point *towards* the endpoint
    far = path[-1]
    direction = np.array([pt[0] - far[0], pt[1] - far[1]], dtype=np.float64)
    norm = np.linalg.norm(direction)
    if norm > 0:
        direction /= norm
    return direction


def _connect_endpoints(skel: np.ndarray, max_gap: int,
                       direction_weight: float = 0.5,
                       look_ahead: int = 6) -> np.ndarray:
    """
    Connect pairs of nearby endpoints in `skel`.

    For each endpoint we:
      • find the nearest unmatched endpoint within `max_gap` pixels
      • optionally bias toward endpoints whose tangent directions
        are roughly collinear (controlled by `direction_weight`)
      • draw a straight line between the matched pair

    Returns a copy of `skel` with the new connections drawn.
    """
    result = skel.copy()
    endpoints = _find_endpoints(skel)

    if len(endpoints) < 2:
        return result

    # Pre-compute tangent directions
    directions = np.array([_direction_at_endpoint(skel, ep, look_ahead)
                           for ep in endpoints])

    # Build KD-tree for fast spatial lookup
    tree = KDTree(endpoints)
    used = set()
    pairs = []

    # Sort candidate pairs by distance so we greedily match closest first
    all_dists = tree.sparse_distance_matrix(tree, max_distance=max_gap,
                                            output_type='coo_matrix')
    # Extract (i, j, dist) triples where i < j
    candidates = []
    for i, j, d in zip(all_dists.row, all_dists.col, all_dists.data):
        if i < j:
            candidates.append((d, i, j))
    candidates.sort()

    for dist, i, j in candidates:
        if i in used or j in used:
            continue

        # Direction compatibility: the two tangent vectors should point
        # roughly toward each other.  Compute a score in [0, 1].
        d_i = directions[i]
        d_j = directions[j]
        if np.linalg.norm(d_i) > 0 and np.linalg.norm(d_j) > 0:
            # Vector from i -> j
            link = endpoints[j] - endpoints[i]
            link_norm = np.linalg.norm(link)
            if link_norm > 0:
                link_unit = link / link_norm
                # endpoint i's tangent should point toward j  (dot > 0)
                # endpoint j's tangent should point toward i  (dot < 0 with link)
                score_i = np.dot(d_i, link_unit)
                score_j = np.dot(d_j, -link_unit)
                compat = (score_i + score_j) / 2.0  # range [-1, 1]
            else:
                compat = 1.0
        else:
            compat = 0.0

        # Accept the pair if direction compatibility is above threshold
        # (lower direction_weight = accept more aggressively)
        threshold = -1.0 + direction_weight * 1.0  # maps [0,1] -> [-1, 0]
        if compat < threshold:
            continue

        used.add(i)
        used.add(j)
        pairs.append((i, j))

    # Draw connecting lines
    for i, j in pairs:
        r0, c0 = int(endpoints[i][0]), int(endpoints[i][1])
        r1, c1 = int(endpoints[j][0]), int(endpoints[j][1])
        rr, cc = sk_line(r0, c0, r1, c1)
        result[rr, cc] = True

    return result


# ──────────────────────────────────────────────
# ComfyUI Node
# ──────────────────────────────────────────────

class ContourGapCloser:
    """
    Joins broken contour lines to produce closed, continuous paths.

    Inputs
    ------
    image : IMAGE
        The sketch / contour image (white background, dark lines or vice-versa).

    Parameters
    ----------
    max_gap : INT
        Maximum pixel distance between two endpoints to consider joining.
    threshold : FLOAT
        Binarisation threshold (0-1).  Pixels darker/lighter than this
        are treated as contour foreground depending on `invert`.
    invert : BOOL
        If True, dark pixels are foreground (typical for black-on-white
        sketches).  If False, bright pixels are foreground.
    morph_close_radius : INT
        Radius of a disk structuring element for an initial morphological
        closing pass.  Set to 0 to skip.
    re_skeletonize : BOOL
        If True, skeletonize the result after gap-filling so lines stay
        single-pixel-wide.
    direction_weight : FLOAT
        How strongly to prefer connecting endpoints whose tangent
        directions are collinear (0 = ignore direction, 1 = strict).
    iterations : INT
        Number of detect-and-connect iterations.  More iterations can
        close chains of small gaps.
    output_mode : ["skeleton", "dilated", "original_thickness"]
        How to render the output contour lines.
    dilation_radius : INT
        Line thickness (radius) when output_mode is "dilated".
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "max_gap": ("INT", {
                    "default": 20,
                    "min": 1,
                    "max": 200,
                    "step": 1,
                    "display": "slider",
                    "tooltip": "Maximum pixel distance between endpoints to bridge",
                }),
                "threshold": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.01,
                    "max": 0.99,
                    "step": 0.01,
                    "display": "slider",
                    "tooltip": "Binarization threshold (0-1)",
                }),
                "invert": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "True = dark lines on light background",
                }),
                "morph_close_radius": ("INT", {
                    "default": 2,
                    "min": 0,
                    "max": 20,
                    "step": 1,
                    "display": "slider",
                    "tooltip": "Morphological closing radius to heal micro-gaps first (0=skip)",
                }),
                "re_skeletonize": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Skeletonize the output to single-pixel lines",
                }),
                "direction_weight": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider",
                    "tooltip": "Direction matching strictness (0=any, 1=strict collinear)",
                }),
                "iterations": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "tooltip": "Repeat endpoint-detection + connection N times",
                }),
                "output_mode": (["original_thickness", "skeleton", "dilated"],),
                "dilation_radius": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "display": "slider",
                    "tooltip": "Line thickness when output_mode='dilated'",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "contour_mask")
    FUNCTION = "close_gaps"
    CATEGORY = "image/contour"
    DESCRIPTION = ("Detects broken contour lines and joins nearby endpoints "
                   "to produce closed, continuous paths.")

    def close_gaps(self, image: torch.Tensor, max_gap: int,
                   threshold: float, invert: bool,
                   morph_close_radius: int, re_skeletonize: bool,
                   direction_weight: float, iterations: int,
                   output_mode: str, dilation_radius: int):

        batch_out_images = []
        batch_out_masks = []

        for b in range(image.shape[0]):
            # ── 1. To numpy grayscale ──
            img_np = image[b].cpu().numpy()  # (H, W, C)  float32 0-1
            if img_np.ndim == 3:
                gray = np.mean(img_np, axis=2)
            else:
                gray = img_np.copy()

            # ── 2. Binarize ──
            binary = _binarize(gray, threshold, invert)

            # ── 3. Optional morphological closing for micro-gaps ──
            if morph_close_radius > 0:
                selem = disk(morph_close_radius)
                binary = closing(binary, selem)

            # ── 4. Skeletonize ──
            skel = skeletonize(binary)

            # ── 5. Iterative endpoint connection ──
            for _ in range(iterations):
                skel = _connect_endpoints(skel, max_gap,
                                          direction_weight=direction_weight)

            # ── 6. Optional re-skeletonize ──
            if re_skeletonize:
                skel = skeletonize(skel)

            # ── 7. Build output ──
            if output_mode == "skeleton":
                out_binary = skel
            elif output_mode == "dilated":
                selem = disk(dilation_radius)
                out_binary = ndimage.binary_dilation(skel, structure=selem)
            else:  # original_thickness
                # Combine original binary with the new skeleton connections
                out_binary = binary | skel

            # Convert to float image (dark lines on white bg if inverted)
            out_float = np.where(out_binary, 0.0, 1.0) if invert else \
                        np.where(out_binary, 1.0, 0.0)
            out_rgb = np.stack([out_float] * 3, axis=-1).astype(np.float32)

            mask_float = out_binary.astype(np.float32)

            batch_out_images.append(torch.from_numpy(out_rgb))
            batch_out_masks.append(torch.from_numpy(mask_float))

        out_image = torch.stack(batch_out_images, dim=0)
        out_mask = torch.stack(batch_out_masks, dim=0)
        return (out_image, out_mask)


# ──────────────────────────────────────────────
# Debug / Visualisation Node
# ──────────────────────────────────────────────

class ContourEndpointVisualizer:
    """
    Diagnostic node: highlights detected endpoints on the image so you
    can tune parameters before running the gap closer.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "threshold": ("FLOAT", {
                    "default": 0.5, "min": 0.01, "max": 0.99, "step": 0.01,
                }),
                "invert": ("BOOLEAN", {"default": True}),
                "morph_close_radius": ("INT", {
                    "default": 2, "min": 0, "max": 20, "step": 1,
                }),
                "endpoint_radius": ("INT", {
                    "default": 5, "min": 1, "max": 20, "step": 1,
                    "tooltip": "Radius of the circles drawn at each endpoint",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("debug_image",)
    FUNCTION = "visualize"
    CATEGORY = "image/contour"
    DESCRIPTION = "Shows detected contour endpoints as red circles for debugging."

    def visualize(self, image, threshold, invert, morph_close_radius,
                  endpoint_radius):
        from skimage.draw import disk as sk_disk

        batch_out = []
        for b in range(image.shape[0]):
            img_np = image[b].cpu().numpy().copy()
            if img_np.ndim == 2:
                img_np = np.stack([img_np] * 3, axis=-1)

            gray = np.mean(img_np, axis=2)
            binary = _binarize(gray, threshold, invert)

            if morph_close_radius > 0:
                selem = disk(morph_close_radius)
                binary = closing(binary, selem)

            skel = skeletonize(binary)
            endpoints = _find_endpoints(skel)

            # Draw red circles at endpoints
            h, w = img_np.shape[:2]
            for ep in endpoints:
                rr, cc = sk_disk((int(ep[0]), int(ep[1])),
                                 endpoint_radius, shape=(h, w))
                img_np[rr, cc] = [1.0, 0.0, 0.0]  # red

            batch_out.append(torch.from_numpy(img_np.astype(np.float32)))

        return (torch.stack(batch_out, dim=0),)