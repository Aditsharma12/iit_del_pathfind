#!/usr/bin/env python3
"""
Urban Mission Planning Challenge — Solution
============================================
Given a high-resolution satellite image and two pixel coordinates,
output a road-constrained path between them.

Strategy:
  1. Load road mask (from ground-truth maps directory when available,
     or extract from satellite image via color thresholding for test data)
  2. Snap start/goal to nearest road pixels
  3. Run Bidirectional A* search on road-pixel graph (8-directional connectivity)
  4. Aggressively simplify path to minimize length (line-of-sight shortcuts)
  5. Output JSON submission

Usage:
  # Evaluate on training data (with known masks):
  python solution.py evaluate --data_dir ump_data

  # Generate submission for test data:
  python solution.py submit --data_dir ump_data --output submission.json

  # Process a single image:
  python solution.py single --image path/to/image.tiff --start X Y --goal X Y

Requirements: numpy, Pillow, tifffile, scipy, opencv-python-headless
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


# ─── JSON Encoder for numpy types ────────────────────────────────────────────

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy integer and float types."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ─── Optional imports with graceful fallback ─────────────────────────────────
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("[WARN] Pillow not installed. Some features may be limited.")

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("[WARN] OpenCV not installed. Using fallback road extraction.")

try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False

try:
    from scipy.ndimage import (
        binary_closing, binary_dilation, label, distance_transform_edt
    )
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("[WARN] SciPy not installed. Some road extraction steps skipped.")


# ─── Image Loading ────────────────────────────────────────────────────────────

def load_image(path: Path) -> np.ndarray:
    """Load a TIFF (or any) image as a numpy array (H, W, C) uint8 RGB."""
    path = Path(path)
    if HAS_TIFFFILE and path.suffix.lower() in ('.tif', '.tiff'):
        img = tifffile.imread(str(path))
        # Handle different shapes
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        elif img.ndim == 3 and img.shape[2] > 3:
            img = img[:, :, :3]
        elif img.ndim == 3 and img.shape[0] in (1, 3, 4):
            # Channel-first format
            img = np.moveaxis(img, 0, -1)
            if img.shape[2] > 3:
                img = img[:, :, :3]
        # Normalize to uint8
        if img.dtype != np.uint8:
            if img.max() > 255:
                img = (img / 256).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
        return img
    elif HAS_PIL:
        return np.array(Image.open(str(path)).convert("RGB"))
    else:
        raise RuntimeError(f"Cannot load image {path}: install tifffile or Pillow")


def load_mask(path: Path) -> np.ndarray:
    """Load a road mask as a boolean numpy array (True = road)."""
    path = Path(path)
    if HAS_PIL:
        img = Image.open(str(path)).convert("L")
        arr = np.array(img)
        return arr > 127
    elif HAS_TIFFFILE:
        arr = tifffile.imread(str(path))
        if arr.ndim == 3:
            arr = arr[..., 0] if arr.shape[-1] in (1,3,4) else arr[0]
        return arr > 127
    else:
        raise RuntimeError(f"Cannot load mask {path}")


# ─── Road Extraction from Satellite Image ────────────────────────────────────

def extract_road_mask_cv2(img: np.ndarray) -> np.ndarray:
    """
    Extract road mask from RGB satellite image using OpenCV.
    Roads in aerial imagery appear as gray/dark-gray asphalt with
    low saturation (R≈G≈B) and medium-low brightness.
    Uses multi-scale approach with morphological refinement.
    """
    # Convert to HSV for saturation-based filtering
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # Roads: low saturation + medium brightness (not too dark/bright)
    # Tuned thresholds based on reference satellite-to-mask analysis
    road_mask_strict = (
        (s < 35) &          # Very low saturation (gray asphalt)
        (v > 50) &          # Not too dark
        (v < 220)           # Not too bright (avoid bright rooftops)
    )

    road_mask_loose = (
        (s < 55) &          # Slightly loosened for varied asphalt types
        (v > 40) &
        (v < 230)
    )

    # Edge detection for road boundary guidance
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 25, 80)

    # Dilate edges to cover road width
    road_edge_guide = cv2.dilate(edges, np.ones((7, 7), np.uint8), iterations=2)

    # Combine strict mask (core roads) with edges to capture road boundaries
    combined = road_mask_strict.astype(np.uint8) * 255
    combined = cv2.bitwise_or(combined, road_edge_guide)

    # Morphological cleanup: close gaps within roads
    kernel_close = np.ones((9, 9), np.uint8)
    closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_close)

    # Open to remove isolated noise blobs
    kernel_open = np.ones((3, 3), np.uint8)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)

    # Additional dilation to ensure connectivity along road widths
    final = cv2.dilate(opened, np.ones((5, 5), np.uint8), iterations=1)

    return final > 0


def extract_road_mask_numpy(img: np.ndarray) -> np.ndarray:
    """Fallback road extractor using pure numpy (no OpenCV)."""
    r, g, b = img[:, :, 0].astype(float), img[:, :, 1].astype(float), img[:, :, 2].astype(float)

    # Grayscale brightness
    bright = (r + g + b) / 3.0

    # Saturation proxy: max - min channel difference
    sat = np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)

    # Road: medium brightness, low saturation
    road_mask = (sat < 40) & (bright > 50) & (bright < 220)

    # Simple dilation
    pad = 5
    h, w = road_mask.shape
    padded = np.pad(road_mask.astype(np.uint8), pad, mode='edge')
    dilated = np.zeros_like(road_mask)
    for dy in range(-pad, pad + 1):
        for dx in range(-pad, pad + 1):
            dilated |= padded[pad + dy:pad + dy + h, pad + dx:pad + dx + w].astype(bool)

    return dilated


def extract_road_mask(img: np.ndarray) -> np.ndarray:
    """Extract road mask from satellite image (automatically picks best method)."""
    if HAS_CV2:
        return extract_road_mask_cv2(img)
    else:
        return extract_road_mask_numpy(img)


# ─── Path Planning ────────────────────────────────────────────────────────────

def snap_to_road(coord: Tuple[int, int], road_mask: np.ndarray,
                 max_dist: int = 150) -> Optional[Tuple[int, int]]:
    """
    Snap a coordinate to the nearest road pixel using distance transform.
    Returns None if no road pixel found within max_dist.
    """
    x, y = coord
    h, w = road_mask.shape

    # Clamp to image bounds
    x = max(0, min(w - 1, x))
    y = max(0, min(h - 1, y))

    if road_mask[y, x]:
        return (x, y)

    # Search in expanding square window
    if HAS_SCIPY:
        dist = distance_transform_edt(~road_mask)
        if dist[y, x] > max_dist:
            return None
        y0, y1 = max(0, y - max_dist), min(h, y + max_dist + 1)
        x0, x1 = max(0, x - max_dist), min(w, x + max_dist + 1)
        local_mask = road_mask[y0:y1, x0:x1]
        local_dist = dist[y0:y1, x0:x1]
        if not local_mask.any():
            return None
        local_dist_on_road = np.where(local_mask, local_dist, np.inf)
        idx = np.unravel_index(np.argmin(local_dist_on_road), local_dist_on_road.shape)
        return (x0 + idx[1], y0 + idx[0])
    else:
        for radius in range(1, max_dist + 1):
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if abs(dx) != radius and abs(dy) != radius:
                        continue
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < w and 0 <= ny < h and road_mask[ny, nx]:
                        return (nx, ny)
        return None


def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    """Euclidean distance heuristic for A*."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def astar(road_mask: np.ndarray, start: Tuple[int, int],
          goal: Tuple[int, int], timeout: float = 60.0) -> Optional[List[Tuple[int, int]]]:
    """
    Bidirectional A* search on road pixel graph (8-directional connectivity).
    Only moves through road pixels.
    Returns list of (x, y) coordinates or None if no path found.
    """
    h, w = road_mask.shape

    # Step costs for 8 directions
    DIRECTIONS = [
        (0, 1, 1.0), (0, -1, 1.0), (1, 0, 1.0), (-1, 0, 1.0),
        (1, 1, 1.4142), (1, -1, 1.4142), (-1, 1, 1.4142), (-1, -1, 1.4142),
    ]

    # Forward search
    open_fwd = []
    heapq.heappush(open_fwd, (heuristic(start, goal), 0.0, start))
    g_fwd = {start: 0.0}
    parent_fwd = {}
    closed_fwd = set()

    # Backward search
    open_bwd = []
    heapq.heappush(open_bwd, (heuristic(goal, start), 0.0, goal))
    g_bwd = {goal: 0.0}
    parent_bwd = {}
    closed_bwd = set()

    best_total = float('inf')
    meeting_node = None
    t_start = time.time()

    while open_fwd or open_bwd:
        if time.time() - t_start > timeout:
            break  # Timeout — return best found so far

        # Expand forward
        if open_fwd:
            f, g, current = heapq.heappop(open_fwd)
            if current not in closed_fwd:
                closed_fwd.add(current)
                cx, cy = current
                for dx, dy, cost in DIRECTIONS:
                    nx, ny = cx + dx, cy + dy
                    if nx < 0 or nx >= w or ny < 0 or ny >= h:
                        continue
                    if not road_mask[ny, nx]:
                        continue
                    neighbor = (nx, ny)
                    if neighbor in closed_fwd:
                        continue
                    tg = g + cost
                    if tg < g_fwd.get(neighbor, float('inf')):
                        g_fwd[neighbor] = tg
                        parent_fwd[neighbor] = current
                        heapq.heappush(open_fwd, (tg + heuristic(neighbor, goal), tg, neighbor))
                    # Check meeting with backward search
                    if neighbor in g_bwd:
                        total = tg + g_bwd[neighbor]
                        if total < best_total:
                            best_total = total
                            meeting_node = neighbor

        # Expand backward
        if open_bwd:
            f, g, current = heapq.heappop(open_bwd)
            if current not in closed_bwd:
                closed_bwd.add(current)
                cx, cy = current
                for dx, dy, cost in DIRECTIONS:
                    nx, ny = cx + dx, cy + dy
                    if nx < 0 or nx >= w or ny < 0 or ny >= h:
                        continue
                    if not road_mask[ny, nx]:
                        continue
                    neighbor = (nx, ny)
                    if neighbor in closed_bwd:
                        continue
                    tg = g + cost
                    if tg < g_bwd.get(neighbor, float('inf')):
                        g_bwd[neighbor] = tg
                        parent_bwd[neighbor] = current
                        heapq.heappush(open_bwd, (tg + heuristic(neighbor, start), tg, neighbor))
                    # Check meeting with forward search
                    if neighbor in g_fwd:
                        total = g_fwd[neighbor] + tg
                        if total < best_total:
                            best_total = total
                            meeting_node = neighbor

        # Check termination: if best seen is better than any possible future
        if meeting_node is not None:
            # Check lower bounds on remaining open nodes
            fwd_min = open_fwd[0][0] if open_fwd else float('inf')
            bwd_min = open_bwd[0][0] if open_bwd else float('inf')
            if fwd_min + bwd_min >= best_total:
                break

    if meeting_node is None:
        # Fall back to standard A* (unidirectional) if bidirectional failed
        return _astar_unidirectional(road_mask, start, goal, timeout)

    # Reconstruct path: forward half
    path_fwd = []
    node = meeting_node
    while node in parent_fwd:
        path_fwd.append(node)
        node = parent_fwd[node]
    path_fwd.append(start)
    path_fwd.reverse()

    # Reconstruct path: backward half
    path_bwd = []
    node = meeting_node
    while node in parent_bwd:
        node = parent_bwd[node]
        path_bwd.append(node)

    return path_fwd + path_bwd


def _astar_unidirectional(road_mask: np.ndarray, start: Tuple[int, int],
                          goal: Tuple[int, int],
                          timeout: float = 60.0) -> Optional[List[Tuple[int, int]]]:
    """Standard unidirectional A* as fallback."""
    h, w = road_mask.shape
    DIRECTIONS = [
        (0, 1, 1.0), (0, -1, 1.0), (1, 0, 1.0), (-1, 0, 1.0),
        (1, 1, 1.4142), (1, -1, 1.4142), (-1, 1, 1.4142), (-1, -1, 1.4142),
    ]

    open_heap = []
    heapq.heappush(open_heap, (heuristic(start, goal), 0.0, start))
    g_score = {start: 0.0}
    came_from = {}
    closed = set()
    t_start = time.time()

    while open_heap:
        if time.time() - t_start > timeout:
            break
        f, g, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        closed.add(current)

        if current == goal:
            path = []
            node = current
            while node in came_from:
                path.append(node)
                node = came_from[node]
            path.append(start)
            path.reverse()
            return path

        cx, cy = current
        for dx, dy, cost in DIRECTIONS:
            nx, ny = cx + dx, cy + dy
            if nx < 0 or nx >= w or ny < 0 or ny >= h:
                continue
            if not road_mask[ny, nx]:
                continue
            neighbor = (nx, ny)
            if neighbor in closed:
                continue
            tentative_g = g + cost
            if tentative_g < g_score.get(neighbor, float('inf')):
                g_score[neighbor] = tentative_g
                came_from[neighbor] = current
                f_new = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_heap, (f_new, tentative_g, neighbor))

    return None


def astar_downsampled(road_mask: np.ndarray, start: Tuple[int, int],
                      goal: Tuple[int, int],
                      scale: int = 2) -> Optional[List[Tuple[int, int]]]:
    """
    Run A* on a downsampled version of the road mask for speed,
    then scale coordinates back to original resolution.
    """
    h, w = road_mask.shape
    new_h, new_w = h // scale, w // scale
    small_mask = road_mask[:new_h * scale, :new_w * scale].reshape(
        new_h, scale, new_w, scale
    ).any(axis=(1, 3))

    small_start = (start[0] // scale, start[1] // scale)
    small_goal = (goal[0] // scale, goal[1] // scale)

    small_start = snap_to_road(small_start, small_mask, max_dist=30)
    small_goal = snap_to_road(small_goal, small_mask, max_dist=30)

    if small_start is None or small_goal is None:
        return None

    small_path = astar(small_mask, small_start, small_goal, timeout=30.0)
    if small_path is None:
        return None

    # Scale back
    full_path = [(x * scale, y * scale) for x, y in small_path]
    return full_path


# ─── Path Simplification ─────────────────────────────────────────────────────

def rasterize_segment(a: Tuple[int, int], b: Tuple[int, int]) -> List[Tuple[int, int]]:
    """Rasterize a line segment (Bresenham-like) into integer pixel coords."""
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    steps = max(abs(dx), abs(dy), 1)
    pts = []
    for i in range(steps + 1):
        t = i / float(steps)
        x = int(round(a[0] + dx * t))
        y = int(round(a[1] + dy * t))
        pts.append((x, y))
    return pts


def segment_on_road(a: Tuple[int, int], b: Tuple[int, int],
                    road_mask: np.ndarray) -> bool:
    """Check if every pixel in the segment lies on the road mask."""
    h, w = road_mask.shape
    for x, y in rasterize_segment(a, b):
        if x < 0 or x >= w or y < 0 or y >= h:
            return False
        if not road_mask[y, x]:
            return False
    return True


def douglas_peucker_road(path: List[Tuple[int, int]],
                         road_mask: np.ndarray,
                         epsilon: float = 2.0) -> List[Tuple[int, int]]:
    """
    Ramer-Douglas-Peucker path simplification that only removes waypoints
    if the shortcut segment stays entirely on the road mask.
    """
    if len(path) <= 2:
        return path

    start, end = path[0], path[-1]
    max_dist = 0.0
    max_idx = 0
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length_sq = dx * dx + dy * dy

    for i in range(1, len(path) - 1):
        px, py = path[i]
        if length_sq == 0:
            dist = math.hypot(px - start[0], py - start[1])
        else:
            t = ((px - start[0]) * dx + (py - start[1]) * dy) / length_sq
            t = max(0.0, min(1.0, t))
            proj_x = start[0] + t * dx
            proj_y = start[1] + t * dy
            dist = math.hypot(px - proj_x, py - proj_y)
        if dist > max_dist:
            max_dist = dist
            max_idx = i

    shortcut_valid = segment_on_road(start, end, road_mask)

    if max_dist <= epsilon and shortcut_valid:
        return [start, end]
    else:
        left = douglas_peucker_road(path[:max_idx + 1], road_mask, epsilon)
        right = douglas_peucker_road(path[max_idx:], road_mask, epsilon)
        return left[:-1] + right


def greedy_shortcut(path: List[Tuple[int, int]],
                    road_mask: np.ndarray) -> List[Tuple[int, int]]:
    """
    Greedy line-of-sight shortcutting: try to skip as many intermediate
    waypoints as possible with direct road segments.
    This is much more aggressive than Douglas-Peucker alone.
    """
    if len(path) <= 2:
        return path

    result = [path[0]]
    i = 0
    while i < len(path) - 1:
        # Binary search for the furthest reachable point
        lo, hi = i + 1, len(path) - 1
        best = i + 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if segment_on_road(path[i], path[mid], road_mask):
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        result.append(path[best])
        i = best

    return result


def simplify_path(path: List[Tuple[int, int]],
                  road_mask: np.ndarray,
                  epsilon: float = 5.0) -> List[Tuple[int, int]]:
    """
    Aggressively simplify path using:
    1. Douglas-Peucker (multiple passes, increasing epsilon)
    2. Greedy line-of-sight shortcuts
    Until convergence.
    """
    if len(path) <= 2:
        return path

    current = path

    # Multiple rounds of Douglas-Peucker with increasing epsilon
    for eps in [2.0, 5.0, 10.0, 20.0]:
        prev_len = len(current) + 1
        for _ in range(3):
            if len(current) == prev_len:
                break
            prev_len = len(current)
            current = douglas_peucker_road(current, road_mask, eps)

    # Greedy line-of-sight pass
    prev_len = len(current) + 1
    for _ in range(5):
        if len(current) == prev_len:
            break
        prev_len = len(current)
        current = greedy_shortcut(current, road_mask)

    # Final DP pass to clean up
    current = douglas_peucker_road(current, road_mask, epsilon)

    if len(current) < 2:
        return [path[0], path[-1]]

    return current


# ─── Fallback: Straight-line path ────────────────────────────────────────────

def straight_line_path(start: Tuple[int, int],
                       goal: Tuple[int, int]) -> List[Tuple[int, int]]:
    """Generate a straight-line path (used as fallback when no road path found)."""
    pts = rasterize_segment(start, goal)
    seen = set()
    result = []
    for p in pts:
        if p not in seen:
            seen.add(p)
            result.append(p)
    return result if len(result) >= 2 else [start, goal]


# ─── Main Pipeline ────────────────────────────────────────────────────────────

def compute_path_for_mission(
    image_path: Path,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    map_path: Optional[Path] = None,
    verbose: bool = True
) -> List[Tuple[int, int]]:
    """
    Full pipeline: load image → extract roads → find path → simplify.
    Returns list of [x, y] waypoints.
    """
    t0 = time.time()

    # Load road mask
    if map_path is not None and map_path.exists():
        if verbose:
            print(f"  Loading road mask: {map_path.name}")
        road_mask = load_mask(map_path)
    else:
        if verbose:
            print(f"  Extracting roads from satellite image: {image_path.name}")
        img = load_image(image_path)
        road_mask = extract_road_mask(img)
        if verbose:
            road_pct = road_mask.sum() / road_mask.size * 100
            print(f"  Road coverage: {road_pct:.1f}%")

    h, w = road_mask.shape

    # Snap start/goal to nearest road pixel
    snapped_start = snap_to_road(start, road_mask, max_dist=150)
    snapped_goal = snap_to_road(goal, road_mask, max_dist=150)

    if snapped_start is None:
        if verbose:
            print(f"  [WARN] Cannot snap start {start} to road. Using original.")
        snapped_start = start
    if snapped_goal is None:
        if verbose:
            print(f"  [WARN] Cannot snap goal {goal} to road. Using original.")
        snapped_goal = goal

    if verbose:
        print(f"  Start: {start} → snapped to {snapped_start}")
        print(f"  Goal:  {goal} → snapped to {snapped_goal}")

    # Run pathfinding
    if verbose:
        print(f"  Running bidirectional A* on {w}×{h} image...")

    path = None

    # For large images, use downsampled A* first, then refine
    if w > 2000 or h > 2000:
        if verbose:
            print(f"  Large image ({w}×{h}): using 2x downsampled A*...")
        path = astar_downsampled(road_mask, snapped_start, snapped_goal, scale=2)
        if path is not None and verbose:
            print(f"  Refining on full resolution...")
        # Refine segments
        if path is not None:
            refined = [path[0]]
            for i in range(1, len(path)):
                local_path = _astar_unidirectional(road_mask, refined[-1], path[i], timeout=10.0)
                if local_path is not None and len(local_path) > 1:
                    refined.extend(local_path[1:])
                else:
                    refined.append(path[i])
            path = refined

    if path is None:
        if verbose:
            print(f"  Running full-resolution bidirectional A*...")
        path = astar(road_mask, snapped_start, snapped_goal, timeout=120.0)

    if path is None:
        if verbose:
            print(f"  [WARN] No road path found! Falling back to straight line.")
        path = straight_line_path(snapped_start, snapped_goal)

    path_len_before = sum(
        math.hypot(path[i+1][0]-path[i][0], path[i+1][1]-path[i][1])
        for i in range(len(path)-1)
    )
    n_pts_before = len(path)

    # Aggressively simplify path
    if verbose:
        print(f"  Simplifying path ({len(path)} waypoints, length={path_len_before:.0f}px)...")
    path = simplify_path(path, road_mask, epsilon=5.0)

    path_len_after = sum(
        math.hypot(path[i+1][0]-path[i][0], path[i+1][1]-path[i][1])
        for i in range(len(path)-1)
    )

    t1 = time.time()
    if verbose:
        print(f"  Done: {n_pts_before} → {len(path)} waypoints, "
              f"length {path_len_before:.0f} → {path_len_after:.0f}px "
              f"({t1-t0:.1f}s)")

    return [[int(x), int(y)] for x, y in path]


def compute_score(path: List[List[int]], road_mask: np.ndarray,
                  img_w: int, img_h: int) -> dict:
    """Compute the score for a path against a road mask."""
    pts = [tuple(p) for p in path]

    length = sum(
        math.hypot(pts[i+1][0]-pts[i][0], pts[i+1][1]-pts[i][1])
        for i in range(len(pts)-1)
    )

    off_pts = 0
    off_segs = 0
    for (x, y) in pts:
        if x < 0 or x >= img_w or y < 0 or y >= img_h or not road_mask[y, x]:
            off_pts += 1
    for i in range(len(pts) - 1):
        seg_pts = rasterize_segment(pts[i], pts[i+1])
        if any(x < 0 or x >= img_w or y < 0 or y >= img_h or not road_mask[y, x]
               for x, y in seg_pts):
            off_segs += 1

    violations = off_pts + off_segs
    score = 1000 - length - 50 * violations
    return {"length": length, "off_pts": off_pts, "off_segs": off_segs,
            "violations": violations, "score": score}


# ─── CLI Interface ────────────────────────────────────────────────────────────

def mode_evaluate(args):
    """Evaluate solution on all training data and print scores."""
    data_dir = Path(args.data_dir)
    metadata_file = data_dir / "reference_metadata.json"

    if not metadata_file.exists():
        print(f"ERROR: {metadata_file} not found")
        sys.exit(1)

    metadata = json.loads(metadata_file.read_text())
    results = []
    total_score = 0.0

    limit = args.limit if hasattr(args, 'limit') and args.limit else len(metadata)
    print(f"Evaluating {limit} training samples...")

    for i, item in enumerate(metadata[:limit]):
        sample_id = item["id"]
        image_path = data_dir / item["image"]
        map_path = data_dir / item["map"]
        start = tuple(item["start"])
        goal = tuple(item["goal"])
        img_size = item["image_size"]  # [width, height]

        print(f"\n[{i+1}/{limit}] {sample_id}")

        try:
            path = compute_path_for_mission(
                image_path, start, goal,
                map_path=map_path,  # Use ground truth for training eval
                verbose=True
            )

            # Score against ground truth mask
            road_mask = load_mask(map_path)
            metrics = compute_score(path, road_mask, img_size[0], img_size[1])
            total_score += metrics["score"]

            print(f"  Score: {metrics['score']:.1f} "
                  f"(length={metrics['length']:.0f}, "
                  f"violations={metrics['violations']})")
            results.append({"id": sample_id, **metrics})

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append({"id": sample_id, "error": str(e)})

    print(f"\n{'='*60}")
    print(f"Total score: {total_score:.1f} across {limit} samples")
    if limit > 0:
        print(f"Average score: {total_score/limit:.1f}")

    if args.output:
        out_data = {"results": results, "total_score": total_score}
        Path(args.output).write_text(json.dumps(out_data, indent=2, cls=NumpyEncoder))
        print(f"Results saved to {args.output}")


def mode_submit(args):
    """
    Generate submission JSON for test images.
    Uses extracted road masks (no ground truth available for test).
    Start/goal coordinates come from mission JSON files or test_metadata.json.
    """
    data_dir = Path(args.data_dir)
    submissions = []

    # Try to load test metadata with start/goal (organizer-provided)
    mission_sources = []

    # Option 1: test_metadata.json (may or may not have start/goal)
    test_meta_file = data_dir / "test_metadata.json"
    if test_meta_file.exists():
        test_meta = json.loads(test_meta_file.read_text())
        for item in test_meta:
            if "start" in item and "goal" in item:
                mission_sources.append(item)
            else:
                # start/goal not in metadata — look for individual mission JSONs
                test_id = item["id"]
                # Search common locations for mission JSON
                search_paths = [
                    data_dir / "test" / f"{test_id}_mission.json",
                    data_dir / "test" / "sats" / f"{test_id}_mission.json",
                    data_dir / f"{test_id}_mission.json",
                ]
                mission_json = None
                for sp in search_paths:
                    if sp.exists():
                        mission_json = sp
                        break

                if mission_json:
                    mission_data = json.loads(mission_json.read_text())
                    item["start"] = mission_data["start"]
                    item["goal"] = mission_data["goal"]
                    item["public_image"] = item.get("public_image",
                        f"test/sats/{test_id}.tiff")
                    mission_sources.append(item)
                else:
                    print(f"[WARN] No start/goal for {test_id}. "
                          f"Using image quarter-point placeholders.")
                    img_size = item.get("image_size", [1500, 1500])
                    item["start"] = [img_size[0] // 4, img_size[1] // 4]
                    item["goal"] = [img_size[0] * 3 // 4, img_size[1] * 3 // 4]
                    item["public_image"] = f"test/sats/{test_id}.tiff"
                    mission_sources.append(item)

    if not mission_sources:
        print("ERROR: No test mission data found.")
        sys.exit(1)

    n = len(mission_sources)
    print(f"Generating submission for {n} test images...")

    for i, item in enumerate(mission_sources):
        test_id = item["id"]
        image_path = data_dir / item["public_image"]
        start = tuple(item["start"])
        goal = tuple(item["goal"])

        print(f"\n[{i+1}/{n}] {test_id}")

        try:
            path = compute_path_for_mission(
                image_path, start, goal,
                map_path=None,  # No ground truth for test
                verbose=True
            )
            submissions.append({"id": test_id, "path": path})
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            # Fallback: straight line between start and goal
            submissions.append({
                "id": test_id,
                "path": [[int(start[0]), int(start[1])],
                         [int(goal[0]), int(goal[1])]]
            })

    # Save submission
    output_path = Path(args.output)
    output_path.write_text(json.dumps(submissions, indent=2, cls=NumpyEncoder))
    print(f"\n✓ Submission saved to {output_path}")
    print(f"  Contains {len(submissions)} entries")


def mode_single(args):
    """Process a single image and print/save the path."""
    image_path = Path(args.image)
    start = tuple(args.start)
    goal = tuple(args.goal)
    map_path = Path(args.map) if hasattr(args, 'map') and args.map else None

    print(f"Processing: {image_path.name}")
    print(f"Start: {start}, Goal: {goal}")

    path = compute_path_for_mission(
        image_path, start, goal,
        map_path=map_path,
        verbose=True
    )

    result = {"id": image_path.stem, "path": path}

    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2, cls=NumpyEncoder))
        print(f"Path saved to {args.output}")
    else:
        print(f"Path ({len(path)} waypoints):")
        show = min(5, len(path))
        for pt in path[:show]:
            print(f"  {pt}")
        if len(path) > show * 2:
            print(f"  ... ({len(path) - show * 2} more) ...")
            for pt in path[-show:]:
                print(f"  {pt}")


def mode_visualize(args):
    """Visualize a path overlaid on the satellite image (requires OpenCV)."""
    if not HAS_CV2:
        print("ERROR: OpenCV required for visualization. Install with: pip install opencv-python")
        sys.exit(1)

    image_path = Path(args.image)
    pred_json = Path(args.pred)

    img = load_image(image_path)
    data = json.loads(pred_json.read_text())

    # Handle both single result and list
    if isinstance(data, list):
        stem = image_path.stem
        match = next((d for d in data if d["id"] == stem), data[0])
    else:
        match = data

    path = match["path"]

    # Draw path on image
    vis = img.copy()
    pts = [(int(x), int(y)) for x, y in path]
    for i in range(len(pts) - 1):
        cv2.line(vis, pts[i], pts[i+1], (255, 0, 0), 2)
    cv2.circle(vis, pts[0], 8, (0, 255, 0), -1)   # Start: green
    cv2.circle(vis, pts[-1], 8, (0, 0, 255), -1)  # Goal: red

    # Optional: overlay road mask
    if hasattr(args, 'map') and args.map:
        road_mask = load_mask(Path(args.map))
        road_overlay = np.zeros_like(vis)
        road_overlay[road_mask] = [0, 100, 0]
        vis = cv2.addWeighted(vis, 0.85, road_overlay, 0.15, 0)

    # Save visualization
    out_path = args.output if args.output else f"{image_path.stem}_path.png"
    out_img = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_path, out_img)
    print(f"Visualization saved to {out_path}")


# ─── Entry Point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Urban Mission Planning Challenge — Solution"
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    # Evaluate mode
    p_eval = sub.add_parser("evaluate", help="Evaluate on training data")
    p_eval.add_argument("--data_dir", default="ump_data",
                        help="Path to ump_data repo directory")
    p_eval.add_argument("--output", help="Save evaluation results to JSON")
    p_eval.add_argument("--limit", type=int, help="Limit to N samples")

    # Submit mode
    p_submit = sub.add_parser("submit", help="Generate submission for test data")
    p_submit.add_argument("--data_dir", default="ump_data",
                           help="Path to ump_data repo directory")
    p_submit.add_argument("--output", default="submission.json",
                           help="Output submission JSON file")

    # Single image mode
    p_single = sub.add_parser("single", help="Process a single image")
    p_single.add_argument("--image", required=True, help="Path to satellite TIFF")
    p_single.add_argument("--start", type=int, nargs=2, metavar=("X", "Y"),
                           required=True, help="Start coordinate [x y]")
    p_single.add_argument("--goal", type=int, nargs=2, metavar=("X", "Y"),
                           required=True, help="Goal coordinate [x y]")
    p_single.add_argument("--map", help="Optional ground truth road mask")
    p_single.add_argument("--output", help="Save path JSON to file")

    # Visualize mode
    p_vis = sub.add_parser("visualize", help="Visualize path on image")
    p_vis.add_argument("--image", required=True, help="Path to satellite TIFF")
    p_vis.add_argument("--pred", required=True, help="Prediction JSON file")
    p_vis.add_argument("--map", help="Optional road mask for overlay")
    p_vis.add_argument("--output", help="Output visualization image path")

    args = parser.parse_args()

    if args.mode == "evaluate":
        mode_evaluate(args)
    elif args.mode == "submit":
        mode_submit(args)
    elif args.mode == "single":
        mode_single(args)
    elif args.mode == "visualize":
        mode_visualize(args)


if __name__ == "__main__":
    main()
