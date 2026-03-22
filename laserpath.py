"""
ComfyUI Custom Node: Laser Cutter Path Tracer
Detects internal contours (holes) in a B&W diagram and traces the optimal
laser cutter head path between them using nearest-neighbor + 2-opt TSP.
"""
 
import numpy as np
import cv2
try:
    import torch
except ImportError:
    torch = None
from itertools import combinations
 
 
# ──────────────────────────────────────────────────────────────────────
#  Geometry helpers
# ──────────────────────────────────────────────────────────────────────
 
def contour_centroid(contour):
    """Return (cx, cy) of a contour using image moments."""
    M = cv2.moments(contour)
    if M["m00"] == 0:
        # fallback: mean of all points
        pts = contour.reshape(-1, 2)
        return float(pts[:, 0].mean()), float(pts[:, 1].mean())
    return float(M["m10"] / M["m00"]), float(M["m01"] / M["m00"])
 
 
def contour_nearest_edge_point(contour, ref_point):
    """Return the point on `contour` closest to `ref_point`."""
    pts = contour.reshape(-1, 2).astype(np.float64)
    ref = np.array(ref_point, dtype=np.float64)
    dists = np.linalg.norm(pts - ref, axis=1)
    return tuple(pts[np.argmin(dists)].astype(int))
 
 
def euclidean(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
 
 
# ──────────────────────────────────────────────────────────────────────
#  TSP solvers
# ──────────────────────────────────────────────────────────────────────
 
def nearest_neighbor_open(points, start_idx, end_idx):
    """
    Greedy nearest-neighbor for an open path.
    Starts at `start_idx`, must finish at `end_idx`.
    Returns ordered list of indices.
    """
    n = len(points)
    if n <= 2:
        return list(range(n))
 
    visited = [False] * n
    path = [start_idx]
    visited[start_idx] = True
    # Reserve end point
    visited[end_idx] = True
 
    current = start_idx
    remaining = n - 2  # exclude start and end
 
    for _ in range(remaining):
        best_dist = float("inf")
        best_idx = -1
        for j in range(n):
            if not visited[j]:
                d = euclidean(points[current], points[j])
                if d < best_dist:
                    best_dist = d
                    best_idx = j
        if best_idx == -1:
            break
        visited[best_idx] = True
        path.append(best_idx)
        current = best_idx
 
    path.append(end_idx)
    return path
 
 
def two_opt_improve(path, points, max_iterations=1000):
    """
    2-opt local search improvement for an open path.
    Keeps first and last elements fixed.
    """
    def path_length(p):
        return sum(euclidean(points[p[i]], points[p[i + 1]]) for i in range(len(p) - 1))
 
    best_dist = path_length(path)
    improved = True
    iteration = 0
 
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        # Only reverse interior segments (keep index 0 and -1 fixed)
        for i in range(1, len(path) - 2):
            for j in range(i + 1, len(path) - 1):
                new_path = path[:i] + path[i : j + 1][::-1] + path[j + 1 :]
                new_dist = path_length(new_path)
                if new_dist < best_dist - 1e-6:
                    path = new_path
                    best_dist = new_dist
                    improved = True
                    break
            if improved:
                break
 
    return path
 
 
def solve_tsp(points, start_idx, end_idx):
    """Solve open-path TSP with fixed start and end using NN + 2-opt."""
    if len(points) <= 1:
        return list(range(len(points)))
    if len(points) == 2:
        return [start_idx, end_idx]
 
    path = nearest_neighbor_open(points, start_idx, end_idx)
    path = two_opt_improve(path, points)
    return path
 
 
# ──────────────────────────────────────────────────────────────────────
#  Contour classification
# ──────────────────────────────────────────────────────────────────────
 
def find_internal_contours(binary_img, min_area=30, max_area_ratio=0.5):
    """
    Detect internal contours (holes) inside the external boundary.
 
    Strategy:
      1. Find all contours with full hierarchy.
      2. Identify the external contour as the largest by area.
      3. Internal contours are contours whose centroid lies inside
         the external contour and whose area is much smaller.
 
    Returns: (external_contour, list_of_internal_contours)
    """
    contours, hierarchy = cv2.findContours(
        binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
 
    if not contours:
        return None, []
 
    # Sort by area descending
    areas = [cv2.contourArea(c) for c in contours]
    max_area = max(areas) if areas else 0
 
    if max_area == 0:
        return None, []
 
    # External contour = largest
    ext_idx = int(np.argmax(areas))
    external_contour = contours[ext_idx]
    ext_area = areas[ext_idx]
 
    # Internal contours: smaller contours whose centroid is inside the external
    candidates = []
    for i, c in enumerate(contours):
        if i == ext_idx:
            continue
        a = areas[i]
        if a < min_area:
            continue
        if a > ext_area * max_area_ratio:
            continue
 
        cx, cy = contour_centroid(c)
        # Check if centroid is inside external contour
        inside = cv2.pointPolygonTest(external_contour, (cx, cy), False)
        if inside >= 0:
            candidates.append((c, cx, cy, a))
 
    # Deduplicate: drawn circle strokes produce two contours (inner/outer)
    # with nearly identical centroids. Keep the larger one per cluster.
    dedup_radius = 15  # px — merge contours with centroids this close
    candidates.sort(key=lambda x: -x[3])  # sort by area descending
    internal = []
    used_centroids = []
    for c, cx, cy, a in candidates:
        duplicate = False
        for ux, uy in used_centroids:
            if ((cx - ux) ** 2 + (cy - uy) ** 2) ** 0.5 < dedup_radius:
                duplicate = True
                break
        if not duplicate:
            internal.append(c)
            used_centroids.append((cx, cy))
 
    return external_contour, internal
 
 
# ──────────────────────────────────────────────────────────────────────
#  Drawing
# ──────────────────────────────────────────────────────────────────────
 
def draw_path(
    image,
    ordered_centroids,
    line_color=(80, 80, 80),
    start_color=(0, 180, 0),
    end_color=(0, 0, 220),
    point_color=(60, 60, 60),
    line_thickness=2,
    point_radius=6,
):
    """Draw the traced path on the image."""
    out = image.copy()
    if len(out.shape) == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
 
    n = len(ordered_centroids)
    if n == 0:
        return out
 
    # Draw lines
    for i in range(n - 1):
        pt1 = (int(ordered_centroids[i][0]), int(ordered_centroids[i][1]))
        pt2 = (int(ordered_centroids[i + 1][0]), int(ordered_centroids[i + 1][1]))
        cv2.line(out, pt1, pt2, line_color, line_thickness, cv2.LINE_AA)
 
    # Draw points
    for i, (cx, cy) in enumerate(ordered_centroids):
        pt = (int(cx), int(cy))
        if i == 0:
            cv2.circle(out, pt, point_radius + 2, start_color, -1, cv2.LINE_AA)
        elif i == n - 1:
            cv2.circle(out, pt, point_radius + 2, end_color, -1, cv2.LINE_AA)
        else:
            cv2.circle(out, pt, point_radius, point_color, -1, cv2.LINE_AA)
 
    return out
 
 
# ──────────────────────────────────────────────────────────────────────
#  ComfyUI Node
# ──────────────────────────────────────────────────────────────────────
 
class LaserPathTracerNode:
    """
    Traces the optimal laser cutter path between internal holes in a
    B&W diagram. Start = bottom-right, End = bottom-left.
    """
 
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "binary_threshold": (
                    "INT",
                    {"default": 128, "min": 0, "max": 255, "step": 1},
                ),
                "min_hole_area": (
                    "INT",
                    {"default": 30, "min": 1, "max": 10000, "step": 1},
                ),
                "max_hole_area_pct": (
                    "FLOAT",
                    {"default": 0.4, "min": 0.01, "max": 0.99, "step": 0.01},
                ),
                "line_thickness": (
                    "INT",
                    {"default": 2, "min": 1, "max": 10, "step": 1},
                ),
                "point_radius": (
                    "INT",
                    {"default": 6, "min": 2, "max": 20, "step": 1},
                ),
                "invert_image": (
                    "BOOLEAN",
                    {"default": False},
                ),
            },
        }
 
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("traced_image", "path_info")
    FUNCTION = "trace_path"
    CATEGORY = "Laser/Path"
 
    def trace_path(
        self,
        image,
        binary_threshold,
        min_hole_area,
        max_hole_area_pct,
        line_thickness,
        point_radius,
        invert_image,
    ):
        # ── 1. Convert ComfyUI image (B,H,W,C float 0-1) to OpenCV ──
        img_np = image[0].cpu().numpy()  # (H, W, C) float 0-1
        img_uint8 = (img_np * 255).clip(0, 255).astype(np.uint8)
 
        if img_uint8.shape[2] == 4:
            img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGBA2BGR)
        else:
            img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
 
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
 
        # ── 2. Binarize ──
        _, binary = cv2.threshold(gray, binary_threshold, 255, cv2.THRESH_BINARY)
        if invert_image:
            binary = cv2.bitwise_not(binary)
 
        # For contour detection we need black shapes on white background.
        # Contours are found on white regions in RETR_TREE with inverted image,
        # or we detect on the inverse. Let's detect contours of the black lines:
        binary_inv = cv2.bitwise_not(binary)
 
        # ── 3. Detect contours ──
        external_contour, internal_contours = find_internal_contours(
            binary_inv,
            min_area=min_hole_area,
            max_area_ratio=max_hole_area_pct,
        )
 
        h, w = gray.shape[:2]
 
        if not internal_contours:
            # Fallback: try detecting on the non-inverted binary
            # (holes as white regions inside a black border)
            external_contour, internal_contours = find_internal_contours(
                binary,
                min_area=min_hole_area,
                max_area_ratio=max_hole_area_pct,
            )
 
        if not internal_contours:
            info = "No internal contours (holes) detected. Try adjusting threshold or min_hole_area."
            out_tensor = torch.from_numpy(img_np).unsqueeze(0)
            return (out_tensor, info)
 
        # ── 4. Compute centroids ──
        centroids = [contour_centroid(c) for c in internal_contours]
 
        # ── 5. Add virtual start (bottom-right) and end (bottom-left) ──
        #    We snap these to the nearest hole edge if within range,
        #    otherwise use corner coordinates offset slightly inward.
        margin_x = int(w * 0.02)
        margin_y = int(h * 0.02)
        start_ref = (w - margin_x, h - margin_y)  # bottom-right
        end_ref = (margin_x, h - margin_y)  # bottom-left
 
        all_points = [start_ref] + centroids + [end_ref]
        start_idx = 0
        end_idx = len(all_points) - 1
 
        # ── 6. Solve TSP ──
        path_order = solve_tsp(all_points, start_idx, end_idx)
        ordered_points = [all_points[i] for i in path_order]
 
        # ── 7. Draw ──
        result_bgr = draw_path(
            img_bgr,
            ordered_points,
            line_thickness=line_thickness,
            point_radius=point_radius,
        )
 
        # ── 8. Convert back to ComfyUI format ──
        result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
        result_float = result_rgb.astype(np.float32) / 255.0
        out_tensor = torch.from_numpy(result_float).unsqueeze(0)
 
        # ── 9. Build info string ──
        total_dist = sum(
            euclidean(ordered_points[i], ordered_points[i + 1])
            for i in range(len(ordered_points) - 1)
        )
        info_lines = [
            f"Holes detected: {len(internal_contours)}",
            f"Path points: {len(ordered_points)} (incl. start & end)",
            f"Total path length: {total_dist:.1f} px",
            f"Start: ({ordered_points[0][0]:.0f}, {ordered_points[0][1]:.0f})  [bottom-right]",
            f"End:   ({ordered_points[-1][0]:.0f}, {ordered_points[-1][1]:.0f})  [bottom-left]",
            "",
            "Visit order (x, y):",
        ]
        for i, pt in enumerate(ordered_points):
            label = ""
            if i == 0:
                label = " <- START"
            elif i == len(ordered_points) - 1:
                label = " <- END"
            info_lines.append(f"  {i}: ({pt[0]:.0f}, {pt[1]:.0f}){label}")
 
        return (out_tensor, "\n".join(info_lines))