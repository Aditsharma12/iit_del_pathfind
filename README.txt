Urban Mission Planning Challenge — README
==========================================

OVERVIEW
----------
This solution finds road-constrained paths between two pixel coordinates
in high-resolution satellite images. Given a satellite TIFF and start/goal
pixel coordinates, it outputs a JSON file with waypoints that travel along roads.

Scoring: 1000 - PathLength - 50 * Violations
(where Violations = off-road segments + out-of-bounds + invalid format)


DEPENDENCIES
--------------
Python 3.8 or newer is required.

Install dependencies:
  pip install -r requirements.txt

Or manually:
  pip install numpy Pillow tifffile scipy opencv-python-headless

For Google Colab:
  !pip install tifffile opencv-python-headless -q
  (numpy, scipy, and Pillow are pre-installed)


HOW TO RUN
----------

1. Clone the dataset repository:
   git clone https://github.com/SamyakSS83/ump_data ump_data

2. Generate submission for test images:
   python solution.py submit --data_dir ump_data --output submission.json

   This outputs submission.json with paths for all 10 test images.
   Note: start/goal coordinates for test images come from the organizer-provided
   mission JSON files (or test_metadata.json if it contains them).

3. Evaluate performance on training data (to verify solution quality):
   python solution.py evaluate --data_dir ump_data --output eval_results.json

   This runs pathfinding on all 50 training images (which have ground truth
   road masks) and prints the score for each one.

   To evaluate on just the first 5 images (faster):
   python solution.py evaluate --data_dir ump_data --limit 5

4. Process a single image:
   python solution.py single \
       --image ump_data/reference/sats/train_001.tiff \
       --start 527 914 \
       --goal 300 1497 \
       --map ump_data/reference/maps/train_001_map.tiff \
       --output my_path.json

5. Visualize a path on the satellite image:
   python solution.py visualize \
       --image ump_data/reference/sats/train_001.tiff \
       --pred my_path.json \
       --map ump_data/reference/maps/train_001_map.tiff \
       --output train_001_visualization.png


ALGORITHM
---------
1. Road Extraction:
   - If a ground truth road mask is available (training data), use it directly.
   - For test data (no mask provided): extract roads from satellite image using:
     * HSV color thresholding: roads are low-saturation (gray asphalt),
       medium brightness pixels
     * Canny edge detection to detect road boundaries
     * Morphological closing/opening to fill gaps and remove noise

2. Pathfinding (A* Search):
   - Treat each road pixel as a graph node
   - 8-directional connectivity (horizontal, vertical, diagonal)
   - A* with Euclidean distance heuristic for efficient search
   - Start/goal coordinates snapped to nearest road pixel (within 100px)

3. Path Simplification (Douglas-Peucker):
   - Reduce waypoints using Ramer-Douglas-Peucker algorithm
   - Only simplifies where the shortcut stays entirely on road
   - Multiple passes until convergence
   - Minimizes PathLength component of the score

4. Fallback:
   - If no road path is found (disconnected graph), outputs a
     straight-line path between start and goal


API KEYS
--------
None required. This solution uses no external APIs or paid services.


EXPECTED RUNTIME
----------------
Per image (1500×1500 pixels):
  - With known road mask (training):  5–60 seconds depending on path complexity
  - With road extraction (test):      10–90 seconds depending on road density

For all 10 test images: approximately 2–15 minutes total.

On Google Colab (T4 GPU, but we use CPU): similar times (A* is CPU-bound).

Tips to speed up:
  - Use --limit flag for quick evaluation on a subset
  - The algorithm automatically uses downsampled pathfinding for very
    large images (>2000px), then refines at full resolution


OUTPUT FORMAT
-------------
submission.json is a JSON array:
[
  {"id": "test_001", "path": [[x1, y1], [x2, y2], ...]},
  {"id": "test_002", "path": [[x1, y1], [x2, y2], ...]},
  ...
]

All coordinates are integers within the image bounds.
Each path contains at least 2 waypoints.


FILES
-----
solution.py       - Main solution script
requirements.txt  - Python dependencies
README.txt        - This file (execution instructions)
submission.json   - Generated submission output


CONTACT
-------
Repository: https://github.com/SamyakSS83/ump_data
Issues:     https://github.com/SamyakSS83/ump_data/issues
