"""
ComfyUI Custom Nodes: Laser Cutter Path Tracer & Hole Counter
==============================================================
Detects ALL internal contours (holes, slots, cutouts, notches, complex
shapes) in a B&W laser-cutting diagram and traces the optimal laser
cutter head path between them.
"""

import math
import numpy as np
import cv2

try:
    import torch
except ImportError:
    torch = None


# ──────────────────────────────────────────────────────────────────────
#  Geometry helpers
# ──────────────────────────────────────────────────────────────────────

def contour_centroid(contour):
    """Return (cx, cy) of a contour using image moments."""
    M = cv2.moments(contour)
    if M["m00"] == 0:
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
#  Contour shape classification
# ──────────────────────────────────────────────────────────────────────

def classify_contour(contour):
    """
    Classify a contour by its geometric properties.
    Returns dict with: type, circularity, aspect_ratio, solidity,
    vertex_count, bbox, area, perimeter.
    """
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    bbox = cv2.boundingRect(contour)

    if perimeter < 1 or area < 1:
        return dict(type="line", circularity=0, aspect_ratio=1,
                    solidity=0, vertex_count=len(contour),
                    bbox=bbox, area=area, perimeter=perimeter)

    circularity = 4 * math.pi * area / (perimeter ** 2)

    rect = cv2.minAreaRect(contour)
    rw, rh = rect[1]
    aspect_ratio = max(rw, rh) / min(rw, rh) if min(rw, rh) > 0 else 1.0

    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0

    epsilon = 0.02 * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, True)
    vertex_count = len(approx)

    # Classification
    if circularity > 0.75:
        shape_type = "hole"
    elif circularity > 0.55 and aspect_ratio < 1.6:
        shape_type = "hole"
    elif aspect_ratio > 2.5:
        shape_type = "slot"
    elif vertex_count == 3:
        shape_type = "triangle"
    elif vertex_count == 4 and solidity > 0.85:
        shape_type = "rectangle"
    elif vertex_count <= 8 and solidity > 0.80:
        shape_type = "polygon"
    elif solidity < 0.60:
        shape_type = "complex"
    else:
        shape_type = "cutout"

    return dict(type=shape_type, circularity=circularity,
                aspect_ratio=aspect_ratio, solidity=solidity,
                vertex_count=vertex_count, bbox=bbox,
                area=area, perimeter=perimeter)


# ──────────────────────────────────────────────────────────────────────
#  Internal feature data structure
# ──────────────────────────────────────────────────────────────────────

class InternalFeature:
    """Represents a single detected internal feature."""
    __slots__ = ("center", "contour", "classification", "bbox", "area")

    def __init__(self, center, contour, classification):
        self.center = center
        self.contour = contour
        self.classification = classification
        self.bbox = classification["bbox"]
        self.area = classification["area"]

    @property
    def type(self):
        return self.classification["type"]

    def info_str(self):
        c = self.classification
        return (
            f"({self.center[0]:.0f}, {self.center[1]:.0f}) "
            f"type={self.type}, area={c['area']:.0f}, "
            f"circ={c['circularity']:.2f}, AR={c['aspect_ratio']:.1f}, "
            f"solid={c['solidity']:.2f}, verts={c['vertex_count']}"
        )


# ──────────────────────────────────────────────────────────────────────
#  TSP solvers
# ──────────────────────────────────────────────────────────────────────

def nearest_neighbor_open(points, start_idx, end_idx):
    n = len(points)
    if n <= 2:
        return list(range(n))
    visited = [False] * n
    path = [start_idx]
    visited[start_idx] = True
    visited[end_idx] = True
    current = start_idx
    for _ in range(n - 2):
        best_dist, best_idx = float("inf"), -1
        for j in range(n):
            if not visited[j]:
                d = euclidean(points[current], points[j])
                if d < best_dist:
                    best_dist, best_idx = d, j
        if best_idx == -1:
            break
        visited[best_idx] = True
        path.append(best_idx)
        current = best_idx
    path.append(end_idx)
    return path


def two_opt_improve(path, points, max_iterations=1000):
    def path_length(p):
        return sum(euclidean(points[p[i]], points[p[i+1]]) for i in range(len(p)-1))
    best_dist = path_length(path)
    improved = True
    iteration = 0
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        for i in range(1, len(path) - 2):
            for j in range(i + 1, len(path) - 1):
                new_path = path[:i] + path[i:j+1][::-1] + path[j+1:]
                new_dist = path_length(new_path)
                if new_dist < best_dist - 1e-6:
                    path, best_dist, improved = new_path, new_dist, True
                    break
            if improved:
                break
    return path


def solve_tsp(points, start_idx, end_idx):
    if len(points) <= 1:
        return list(range(len(points)))
    if len(points) == 2:
        return [start_idx, end_idx]
    path = nearest_neighbor_open(points, start_idx, end_idx)
    path = two_opt_improve(path, points)
    return path


# ──────────────────────────────────────────────────────────────────────
#  Binarization helper
# ──────────────────────────────────────────────────────────────────────

def _binarize(gray_img, binary_threshold, invert_image,
              use_adaptive_threshold, adaptive_block_size, adaptive_c):
    """Return binary image (white bg, black lines)."""
    if use_adaptive_threshold:
        block = adaptive_block_size | 1  # ensure odd
        binary = cv2.adaptiveThreshold(
            gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, block, adaptive_c)
    else:
        _, binary = cv2.threshold(gray_img, binary_threshold, 255,
                                  cv2.THRESH_BINARY)
    if invert_image:
        binary = cv2.bitwise_not(binary)
    return binary


def _find_external_boundary(contours, areas):
    if not contours:
        return -1, None, 0
    ext_idx = int(np.argmax(areas))
    return ext_idx, contours[ext_idx], areas[ext_idx]


def _find_all_pieces(contours, areas, hierarchy, img_area,
                     min_piece_pct=0.002):
    """
    Identify all separate pieces (external boundaries) in the image.

    A piece is a contour that:
      - Has area > min_piece_pct of the total image area
      - Is NOT fully contained inside another larger piece
        (i.e. it's not a hole/feature of a bigger piece)

    Returns list of (contour_index, contour, area) tuples.
    """
    if not contours:
        return []

    min_piece_area = img_area * min_piece_pct
    h = hierarchy[0] if hierarchy is not None else None

    # Gather candidate pieces sorted by area descending
    candidates = []
    for i, c in enumerate(contours):
        a = areas[i]
        if a >= min_piece_area:
            candidates.append((i, c, a))
    candidates.sort(key=lambda x: -x[2])

    if not candidates:
        return []

    # Filter: reject contours whose centroid is inside a larger contour
    # AND whose area is less than half of that larger contour (= a feature).
    # Keep contours that are independent pieces or nearly as large as
    # their parent (= outer/inner stroke edge of the same piece).
    pieces = []
    piece_contours = []  # contours already accepted as pieces

    for idx, c, a in candidates:
        cx, cy = contour_centroid(c)
        is_sub_feature = False

        for p_idx, p_c, p_a in piece_contours:
            if a > p_a * 0.85:
                # Nearly same size as parent — likely inner/outer stroke
                # edge of the same piece. Skip (dedup will handle).
                continue
            inside = cv2.pointPolygonTest(p_c, (cx, cy), False)
            if inside >= 0 and a < p_a * 0.5:
                is_sub_feature = True
                break

        if not is_sub_feature:
            pieces.append((idx, c, a))
            piece_contours.append((idx, c, a))

    return pieces


# ──────────────────────────────────────────────────────────────────────
#  Legacy: simple contour detection (used by HoleCounter pass 3)
# ──────────────────────────────────────────────────────────────────────

def find_internal_contours(binary_img, min_area=30, max_area_ratio=0.5):
    contours, hierarchy = cv2.findContours(
        binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, []
    areas = [cv2.contourArea(c) for c in contours]
    if max(areas) == 0:
        return None, []
    ext_idx = int(np.argmax(areas))
    external_contour = contours[ext_idx]
    ext_area = areas[ext_idx]

    candidates = []
    for i, c in enumerate(contours):
        if i == ext_idx:
            continue
        a = areas[i]
        if a < min_area or a > ext_area * max_area_ratio:
            continue
        cx, cy = contour_centroid(c)
        if cv2.pointPolygonTest(external_contour, (cx, cy), False) >= 0:
            candidates.append((c, cx, cy, a))

    candidates.sort(key=lambda x: -x[3])
    internal, used = [], []
    for c, cx, cy, a in candidates:
        if not any(((cx-ux)**2+(cy-uy)**2)**0.5 < 15 for ux, uy in used):
            internal.append(c)
            used.append((cx, cy))
    return external_contour, internal


# ──────────────────────────────────────────────────────────────────────
#  Pass 1: Contour-based detection (finds ANY drawn shape)
# ──────────────────────────────────────────────────────────────────────

def _contour_based_detect(binary, min_area, max_area_ratio):
    """
    Hierarchy-aware contour detection that finds ALL internal features
    across ALL pieces in the image.
    """
    lines_mask = cv2.bitwise_not(binary)
    contours, hierarchy = cv2.findContours(
        lines_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours or hierarchy is None:
        return [], []

    areas = [cv2.contourArea(c) for c in contours]
    h, w = binary.shape[:2]
    img_area = h * w

    # Find all pieces (external boundaries)
    pieces = _find_all_pieces(contours, areas, hierarchy, img_area)
    if not pieces:
        return [], []

    piece_indices = set(idx for idx, _, _ in pieces)

    features = []
    for i, c in enumerate(contours):
        if i in piece_indices:
            continue
        a = areas[i]
        if a < min_area:
            continue

        cx, cy = contour_centroid(c)

        # Check if this contour is inside ANY piece
        parent_piece = None
        for p_idx, p_c, p_a in pieces:
            if a > p_a * max_area_ratio:
                continue
            if cv2.pointPolygonTest(p_c, (cx, cy), False) >= 0:
                parent_piece = (p_idx, p_c, p_a)
                break

        if parent_piece is None:
            continue

        p_idx, p_c, p_a = parent_piece

        # Boundary-proximity filter (shape-aware)
        cls = classify_contour(c)

        pts = c.reshape(-1, 2).astype(np.float64)
        if len(pts) > 0:
            dists = [abs(cv2.pointPolygonTest(p_c, (float(p[0]), float(p[1])), True))
                     for p in pts[::max(1, len(pts)//20)]]
            median_dist = float(np.median(dists))
            if median_dist < 15:
                if cls["circularity"] < 0.65:
                    continue
                if cls["solidity"] > 0.95 and a > p_a * 0.01:
                    continue

        features.append(InternalFeature((cx, cy), c, cls))

    piece_contours = [p_c for _, p_c, _ in pieces]
    return features, piece_contours


# ──────────────────────────────────────────────────────────────────────
#  Pass 2/3: Flood-fill detection (finds enclosed white pockets)
# ──────────────────────────────────────────────────────────────────────

def _flood_fill_detect_features(binary, min_area, max_area_ratio,
                                morph_close_size=0, morph_dilate_size=0):
    h, w = binary.shape[:2]
    img_area = h * w
    lines_mask = cv2.bitwise_not(binary)

    if morph_close_size > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (morph_close_size, morph_close_size))
        lines_mask = cv2.morphologyEx(lines_mask, cv2.MORPH_CLOSE, k)
    if morph_dilate_size > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (morph_dilate_size, morph_dilate_size))
        lines_mask = cv2.dilate(lines_mask, k, iterations=1)

    repaired = cv2.bitwise_not(lines_mask)
    contours_all, hierarchy_all = cv2.findContours(
        lines_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours_all:
        return []
    areas_all = [cv2.contourArea(c) for c in contours_all]

    pieces = _find_all_pieces(contours_all, areas_all, hierarchy_all, img_area)
    if not pieces:
        return []

    flood = repaired.copy()
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 0)

    num_labels, labels, stats, centroids_cc = cv2.connectedComponentsWithStats(
        flood, connectivity=8)

    features = []
    for lid in range(1, num_labels):
        area = stats[lid, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        cx, cy = float(centroids_cc[lid][0]), float(centroids_cc[lid][1])

        # Find parent piece
        parent_piece = None
        for p_idx, p_c, p_a in pieces:
            if area > p_a * max_area_ratio:
                continue
            if cv2.pointPolygonTest(p_c, (cx, cy), False) >= 0:
                parent_piece = (p_idx, p_c, p_a)
                break
        if parent_piece is None:
            continue

        p_idx, p_c, p_a = parent_piece
        dist_to_boundary = abs(cv2.pointPolygonTest(p_c, (cx, cy), True))
        cmask = (labels == lid).astype(np.uint8) * 255
        cc, _ = cv2.findContours(cmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cont = cc[0] if cc else None
        cls = classify_contour(cont) if cont is not None else dict(
            type="hole", circularity=0, aspect_ratio=1, solidity=0,
            vertex_count=0, bbox=(int(cx), int(cy), 1, 1),
            area=area, perimeter=0)

        if dist_to_boundary < 20:
            if cls["circularity"] < 0.65:
                continue
            if cls["solidity"] > 0.95 and area > p_a * 0.01:
                continue

        features.append(InternalFeature((cx, cy), cont, cls))
    return features


# ──────────────────────────────────────────────────────────────────────
#  Pass 4: Open contour detection (arcs, grooves, engravings)
# ──────────────────────────────────────────────────────────────────────

def _open_contour_detect(binary, piece_contours, min_area, max_area_ratio):
    """
    Detect open contours (arcs, grooves) inside any piece.
    """
    lines_mask = cv2.bitwise_not(binary)
    num_labels, labels, stats, centroids_cc = cv2.connectedComponentsWithStats(
        lines_mask, connectivity=8)

    if num_labels < 2:
        return []

    # Skip the largest line-pixel component (external boundary)
    all_areas = [stats[lid, cv2.CC_STAT_AREA] for lid in range(1, num_labels)]
    max_line_comp_label = int(np.argmax(all_areas)) + 1

    features = []
    for lid in range(1, num_labels):
        if lid == max_line_comp_label:
            continue
        area = stats[lid, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        cx, cy = float(centroids_cc[lid][0]), float(centroids_cc[lid][1])

        # Find parent piece
        parent_piece = None
        for p_c in piece_contours:
            p_a = cv2.contourArea(p_c)
            if area > p_a * max_area_ratio:
                continue
            if cv2.pointPolygonTest(p_c, (cx, cy), False) >= 0:
                parent_piece = p_c
                break
        if parent_piece is None:
            continue

        dist_to_boundary = abs(cv2.pointPolygonTest(parent_piece, (cx, cy), True))
        if dist_to_boundary < 20:
            cmask_pre = (labels == lid).astype(np.uint8) * 255
            cc_pre, _ = cv2.findContours(cmask_pre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cont_pre = cc_pre[0] if cc_pre else None
            if cont_pre is not None:
                cls_pre = classify_contour(cont_pre)
                if cls_pre["circularity"] < 0.65:
                    continue
            else:
                continue

        cmask = (labels == lid).astype(np.uint8) * 255
        cc, _ = cv2.findContours(cmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cont = cc[0] if cc else None
        if cont is not None:
            cls = classify_contour(cont)
            peri = cv2.arcLength(cont, True)
            if peri > 0 and area / peri < 2.0:
                cls["type"] = "arc"
        else:
            cls = dict(type="arc", circularity=0, aspect_ratio=1,
                       solidity=0, vertex_count=0,
                       bbox=(int(cx), int(cy), 1, 1),
                       area=area, perimeter=0)
        features.append(InternalFeature((cx, cy), cont, cls))
    return features


# ──────────────────────────────────────────────────────────────────────
#  Deduplication
# ──────────────────────────────────────────────────────────────────────

def _deduplicate_features(features, dedup_radius):
    if dedup_radius <= 0 or not features:
        return features
    features.sort(key=lambda f: -f.area)
    deduped, used = [], []
    for f in features:
        cx, cy = f.center
        if not any(((cx-ux)**2+(cy-uy)**2)**0.5 < dedup_radius
                   for ux, uy in used):
            deduped.append(f)
            used.append((cx, cy))
    return deduped


# ──────────────────────────────────────────────────────────────────────
#  Master detection: robust_find_features
# ──────────────────────────────────────────────────────────────────────

def robust_find_features(
    gray_img, binary_threshold, invert_image,
    morph_close_size, morph_dilate_size,
    min_feature_area, max_feature_area_pct, dedup_radius,
    use_adaptive_threshold, adaptive_block_size, adaptive_c,
    detect_open_contours=True,
):
    """
    Robustly detect ALL internal features — holes, slots, cutouts,
    polygons, arcs, complex shapes — inside the external boundary.

    Four-pass strategy:
      Pass 1 — Contour-based: any drawn shape inside the boundary.
      Pass 2 — Flood-fill clean: enclosed white pockets.
      Pass 3 — Flood-fill + morph: broken/gapped contours repaired.
      Pass 4 — Open contour: arcs, grooves, engravings.

    Returns: list of InternalFeature objects.
    """
    binary = _binarize(gray_img, binary_threshold, invert_image,
                       use_adaptive_threshold, adaptive_block_size, adaptive_c)

    # Pass 1: contour-based
    features1, piece_contours = _contour_based_detect(
        binary, min_feature_area, max_feature_area_pct)

    # Pass 2: flood-fill clean
    features2 = _flood_fill_detect_features(
        binary, min_feature_area, max_feature_area_pct)

    # Pass 3: flood-fill with morph repair
    features3 = _flood_fill_detect_features(
        binary, min_feature_area, max_feature_area_pct,
        morph_close_size, morph_dilate_size)

    # Pass 4: open contours
    features4 = []
    if detect_open_contours and piece_contours:
        features4 = _open_contour_detect(
            binary, piece_contours,
            min_feature_area, max_feature_area_pct)

    all_features = features1 + features2 + features3 + features4
    return _deduplicate_features(all_features, dedup_radius)


# ──────────────────────────────────────────────────────────────────────
#  Legacy wrapper: robust_find_holes (for HoleCounterNode)
# ──────────────────────────────────────────────────────────────────────

def robust_find_holes(
    gray_img, binary_threshold, invert_image,
    morph_close_size, morph_dilate_size,
    min_hole_area, max_hole_area_pct, dedup_radius,
    use_adaptive_threshold, adaptive_block_size, adaptive_c,
):
    features = robust_find_features(
        gray_img=gray_img, binary_threshold=binary_threshold,
        invert_image=invert_image,
        morph_close_size=morph_close_size,
        morph_dilate_size=morph_dilate_size,
        min_feature_area=min_hole_area,
        max_feature_area_pct=max_hole_area_pct,
        dedup_radius=dedup_radius,
        use_adaptive_threshold=use_adaptive_threshold,
        adaptive_block_size=adaptive_block_size,
        adaptive_c=adaptive_c,
        detect_open_contours=False,
    )
    centers = [f.center for f in features]
    contours = [f.contour for f in features]
    debug = _binarize(gray_img, binary_threshold, invert_image,
                      use_adaptive_threshold, adaptive_block_size, adaptive_c)
    return centers, contours, debug


# ──────────────────────────────────────────────────────────────────────
#  Drawing
# ──────────────────────────────────────────────────────────────────────

FEATURE_COLORS = {
    "hole":      (180, 120,   0),
    "slot":      (0,   160, 160),
    "rectangle": (0,   140,   0),
    "triangle":  (0,   100, 200),
    "polygon":   (160,  80, 160),
    "complex":   (0,    80, 200),
    "cutout":    (100, 140,   0),
    "arc":       (100, 100, 100),
    "line":      (100, 100, 100),
}


def draw_path(image, ordered_centroids, line_color=(80,80,80),
              start_color=(0,180,0), end_color=(0,0,220),
              point_color=(60,60,60), line_thickness=2, point_radius=6):
    out = image.copy()
    if len(out.shape) == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    n = len(ordered_centroids)
    if n == 0:
        return out
    for i in range(n - 1):
        pt1 = (int(ordered_centroids[i][0]), int(ordered_centroids[i][1]))
        pt2 = (int(ordered_centroids[i+1][0]), int(ordered_centroids[i+1][1]))
        cv2.line(out, pt1, pt2, line_color, line_thickness, cv2.LINE_AA)
    for i, (cx, cy) in enumerate(ordered_centroids):
        pt = (int(cx), int(cy))
        if i == 0:
            cv2.circle(out, pt, point_radius+2, start_color, -1, cv2.LINE_AA)
        elif i == n - 1:
            cv2.circle(out, pt, point_radius+2, end_color, -1, cv2.LINE_AA)
        else:
            cv2.circle(out, pt, point_radius, point_color, -1, cv2.LINE_AA)
    return out


def draw_features_path(image, ordered_features, start_point, end_point,
                       line_color=(80,80,80), start_color=(0,180,0),
                       end_color=(0,0,220), line_thickness=2,
                       point_radius=6, show_contours=True,
                       show_labels=True):
    """Draw path with feature-type-aware color-coded annotations."""
    out = image.copy()
    if len(out.shape) == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

    all_pts = [start_point] + [f.center for f in ordered_features] + [end_point]
    for i in range(len(all_pts) - 1):
        p1 = (int(all_pts[i][0]), int(all_pts[i][1]))
        p2 = (int(all_pts[i+1][0]), int(all_pts[i+1][1]))
        cv2.line(out, p1, p2, line_color, line_thickness, cv2.LINE_AA)

    cv2.circle(out, (int(start_point[0]), int(start_point[1])),
               point_radius+2, start_color, -1, cv2.LINE_AA)
    cv2.circle(out, (int(end_point[0]), int(end_point[1])),
               point_radius+2, end_color, -1, cv2.LINE_AA)

    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, feat in enumerate(ordered_features):
        color = FEATURE_COLORS.get(feat.type, (100, 100, 100))
        pt = (int(feat.center[0]), int(feat.center[1]))

        if show_contours and feat.contour is not None:
            cv2.drawContours(out, [feat.contour], -1, color, 2, cv2.LINE_AA)

        cv2.circle(out, pt, point_radius, color, -1, cv2.LINE_AA)
        cv2.circle(out, pt, point_radius, (255, 255, 255), 1, cv2.LINE_AA)

        if show_labels:
            label = f"{i+1}:{feat.type}"
            tsz = cv2.getTextSize(label, font, 0.38, 1)[0]
            tx = int(feat.center[0] - tsz[0] / 2)
            ty = int(feat.center[1] - point_radius - 6)
            cv2.rectangle(out, (tx-2, ty-tsz[1]-2), (tx+tsz[0]+2, ty+4),
                          (255, 255, 255), -1)
            cv2.putText(out, label, (tx, ty), font, 0.38, color, 1, cv2.LINE_AA)
    return out


def draw_hole_annotations(image, hole_centers, hole_contours, point_radius=8):
    out = image.copy()
    if len(out.shape) == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    if not hole_centers:
        return out
    for cont in hole_contours:
        if cont is not None:
            cv2.drawContours(out, [cont], -1, (0, 180, 0), 2, cv2.LINE_AA)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, (cx, cy) in enumerate(hole_centers):
        pt = (int(cx), int(cy))
        cv2.circle(out, pt, point_radius, (0, 0, 220), -1, cv2.LINE_AA)
        cv2.circle(out, pt, point_radius, (255, 255, 255), 1, cv2.LINE_AA)
        label = str(i + 1)
        tsz = cv2.getTextSize(label, font, 0.45, 1)[0]
        tx, ty = int(cx - tsz[0]/2), int(cy - point_radius - 6)
        cv2.rectangle(out, (tx-2, ty-tsz[1]-2), (tx+tsz[0]+2, ty+4),
                      (255, 255, 255), -1)
        cv2.putText(out, label, (tx, ty), font, 0.45, (0, 0, 180), 1, cv2.LINE_AA)
    return out


# ──────────────────────────────────────────────────────────────────────
#  ComfyUI Node — Laser Path Tracer
# ──────────────────────────────────────────────────────────────────────

class LaserPathTracerNode:
    """
    Traces the optimal laser cutter path between ALL internal features
    (holes, slots, cutouts, polygons, arcs, complex shapes).
    Start = bottom-right, End = bottom-left.
    Bypasses gracefully if no features are detected.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "binary_threshold": ("INT", {"default": 128, "min": 0, "max": 255, "step": 1}),
            "min_feature_area": ("INT", {"default": 30, "min": 1, "max": 50000, "step": 1,
                "tooltip": "Minimum contour area (px2) to count as a feature."}),
            "max_feature_area_pct": ("FLOAT", {"default": 0.4, "min": 0.01, "max": 0.99, "step": 0.01,
                "tooltip": "Max feature area as fraction of external boundary."}),
            "morph_close_size": ("INT", {"default": 5, "min": 0, "max": 31, "step": 2,
                "tooltip": "Morphological close kernel. Bridges gaps. 0=off."}),
            "morph_dilate_size": ("INT", {"default": 0, "min": 0, "max": 15, "step": 1,
                "tooltip": "Dilation kernel. Thickens thin strokes. 0=off."}),
            "dedup_radius": ("INT", {"default": 15, "min": 0, "max": 100, "step": 1,
                "tooltip": "Merge features whose centers are within this many px."}),
            "invert_image": ("BOOLEAN", {"default": False}),
            "use_adaptive_threshold": ("BOOLEAN", {"default": False,
                "tooltip": "Adaptive thresholding for noisy/low-contrast images."}),
            "adaptive_block_size": ("INT", {"default": 51, "min": 3, "max": 201, "step": 2}),
            "adaptive_c": ("INT", {"default": 10, "min": -30, "max": 60, "step": 1}),
            "detect_open_contours": ("BOOLEAN", {"default": True,
                "tooltip": "Also detect arcs, grooves, engravings (open line segments)."}),
        }}

    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("traced_image", "feature_count", "path_info")
    FUNCTION = "trace_path"
    CATEGORY = "Laser/Path"

    def trace_path(self, image, binary_threshold, min_feature_area,
                   max_feature_area_pct, morph_close_size, morph_dilate_size,
                   dedup_radius, invert_image,
                   use_adaptive_threshold, adaptive_block_size, adaptive_c,
                   detect_open_contours):

        img_np = image[0].cpu().numpy()
        img_uint8 = (img_np * 255).clip(0, 255).astype(np.uint8)
        if img_uint8.shape[2] == 4:
            img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGBA2BGR)
        else:
            img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]

        # Detect all internal features
        features = robust_find_features(
            gray_img=gray, binary_threshold=binary_threshold,
            invert_image=invert_image,
            morph_close_size=morph_close_size,
            morph_dilate_size=morph_dilate_size,
            min_feature_area=min_feature_area,
            max_feature_area_pct=max_feature_area_pct,
            dedup_radius=dedup_radius,
            use_adaptive_threshold=use_adaptive_threshold,
            adaptive_block_size=adaptive_block_size,
            adaptive_c=adaptive_c,
            detect_open_contours=detect_open_contours)

        feat_count = len(features)

        # Bypass if nothing found
        if feat_count == 0:
            info = (
                "STATUS: BYPASS -- no internal features detected.\n\n"
                "Image returned unchanged. Troubleshooting:\n"
                "  - Toggle invert_image\n"
                "  - Lower min_feature_area\n"
                "  - Enable use_adaptive_threshold\n"
                "  - Increase morph_close_size (7-11)\n"
                "  - Set morph_dilate_size to 2-3")
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(rgb.astype(np.float32) / 255.0).unsqueeze(0)
            return (t, 0, info)

        # TSP path
        margin_x, margin_y = int(w * 0.02), int(h * 0.02)
        start = (w - margin_x, h - margin_y)
        end = (margin_x, h - margin_y)
        centers = [f.center for f in features]
        all_pts = [start] + centers + [end]
        order = solve_tsp(all_pts, 0, len(all_pts) - 1)

        ordered_feats = [features[idx - 1] for idx in order
                         if idx != 0 and idx != len(all_pts) - 1]
        ordered_pts = [all_pts[i] for i in order]

        # Draw path lines only — no annotations
        result_bgr = img_bgr.copy()
        all_centers = [start] + [f.center for f in ordered_feats] + [end]
        for i in range(len(all_centers) - 1):
            pt1 = (int(all_centers[i][0]), int(all_centers[i][1]))
            pt2 = (int(all_centers[i+1][0]), int(all_centers[i+1][1]))
            cv2.line(result_bgr, pt1, pt2, (0, 0, 0), 1, cv2.LINE_AA)

        rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
        out = torch.from_numpy(rgb.astype(np.float32) / 255.0).unsqueeze(0)

        # Info string
        total_dist = sum(euclidean(ordered_pts[i], ordered_pts[i+1])
                         for i in range(len(ordered_pts)-1))
        tc = {}
        for f in features:
            tc[f.type] = tc.get(f.type, 0) + 1

        lines = [
            f"STATUS: OK -- path traced successfully.",
            f"Features detected: {feat_count}",
            f"Breakdown: {', '.join(f'{v}x {k}' for k, v in sorted(tc.items()))}",
            f"Path points: {len(ordered_pts)} (incl. start & end)",
            f"Total path length: {total_dist:.1f} px",
            f"Start: ({start[0]:.0f}, {start[1]:.0f})  [bottom-right]",
            f"End:   ({end[0]:.0f}, {end[1]:.0f})  [bottom-left]",
            "", "Visit order:",
            f"  0: ({start[0]:.0f}, {start[1]:.0f}) <- START",
        ]
        for i, feat in enumerate(ordered_feats):
            lines.append(f"  {i+1}: {feat.info_str()}")
        lines.append(f"  {len(ordered_feats)+1}: ({end[0]:.0f}, {end[1]:.0f}) <- END")

        return (out, feat_count, "\n".join(lines))


# ──────────────────────────────────────────────────────────────────────
#  ComfyUI Node — Hole Counter (backward-compatible)
# ──────────────────────────────────────────────────────────────────────

class HoleCounterNode:
    """Robustly counts internal holes in a B&W laser-cut diagram."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "binary_threshold": ("INT", {"default": 128, "min": 0, "max": 255, "step": 1}),
            "min_hole_area": ("INT", {"default": 50, "min": 1, "max": 50000, "step": 1}),
            "max_hole_area_pct": ("FLOAT", {"default": 0.4, "min": 0.01, "max": 0.99, "step": 0.01}),
            "morph_close_size": ("INT", {"default": 5, "min": 0, "max": 31, "step": 2}),
            "morph_dilate_size": ("INT", {"default": 0, "min": 0, "max": 15, "step": 1}),
            "dedup_radius": ("INT", {"default": 15, "min": 0, "max": 100, "step": 1}),
            "invert_image": ("BOOLEAN", {"default": False}),
            "use_adaptive_threshold": ("BOOLEAN", {"default": False}),
            "adaptive_block_size": ("INT", {"default": 51, "min": 3, "max": 201, "step": 2}),
            "adaptive_c": ("INT", {"default": 10, "min": -30, "max": 60, "step": 1}),
            "annotate_image": ("BOOLEAN", {"default": True}),
            "show_debug_binary": ("BOOLEAN", {"default": False}),
        }}

    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("output_image", "hole_count", "hole_info")
    FUNCTION = "count_holes"
    CATEGORY = "Laser/Path"

    def count_holes(self, image, binary_threshold, min_hole_area,
                    max_hole_area_pct, morph_close_size, morph_dilate_size,
                    dedup_radius, invert_image, use_adaptive_threshold,
                    adaptive_block_size, adaptive_c,
                    annotate_image, show_debug_binary):

        img_np = image[0].cpu().numpy()
        img_uint8 = (img_np * 255).clip(0, 255).astype(np.uint8)
        if img_uint8.shape[2] == 4:
            img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGBA2BGR)
        else:
            img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        centers, contours, debug = robust_find_holes(
            gray, binary_threshold, invert_image,
            morph_close_size, morph_dilate_size,
            min_hole_area, max_hole_area_pct, dedup_radius,
            use_adaptive_threshold, adaptive_block_size, adaptive_c)

        count = len(centers)
        if show_debug_binary:
            vis = cv2.cvtColor(debug, cv2.COLOR_GRAY2BGR)
        elif annotate_image:
            vis = draw_hole_annotations(img_bgr, centers, contours)
        else:
            vis = img_bgr.copy()

        rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        out = torch.from_numpy(rgb.astype(np.float32) / 255.0).unsqueeze(0)

        indexed = sorted(enumerate(centers), key=lambda x: (x[1][1], x[1][0]))
        lines = [f"Holes detected: {count}", ""]
        for rank, (_, (cx, cy)) in enumerate(indexed):
            lines.append(f"  #{rank+1}: ({cx:.0f}, {cy:.0f})")
        return (out, count, "\n".join(lines))


# ──────────────────────────────────────────────────────────────────────
