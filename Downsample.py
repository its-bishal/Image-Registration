

import json
import logging
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s │ %(message)s")
log = logging.getLogger(__name__)


def stitch_downsampled(metadata_path: Path, output_path: Path, scale: float = 0.1):
    with open(metadata_path) as f:
        meta = json.load(f)

    orig_cw = meta["canvas_width"]
    orig_ch = meta["canvas_height"]

    # 1. Calculate new canvas dimensions based on the scale
    new_cw = int(round(orig_cw * scale))
    new_ch = int(round(orig_ch * scale))
    
    log.info(f"Original canvas: {orig_cw}x{orig_ch}")
    log.info(f"Downsampled canvas ({scale*100}%): {new_cw}x{new_ch}")

    # 2. Initialize a blank canvas
    canvas = np.zeros((new_ch, new_cw, 3), dtype=np.uint8)

    # 3. Process and stitch each cell
    for cell in meta["cells"]:
        img_path = cell["path"]
        img = cv2.imread(img_path)
        
        if img is None:
            log.warning(f"Could not read {img_path}")
            continue

        # Original absolute bounding box [x, y, w, h]
        x_orig, y_orig, w_orig, h_orig = cell["bbox"]
        
        # We calculate the absolute bounding coordinates of the cell on the new canvas.
        # Note: We calculate x1/y1 BEFORE figuring out target_w/target_h. 
        x0 = int(round(x_orig * scale))
        y0 = int(round(y_orig * scale))
        x1 = int(round((x_orig + w_orig) * scale))
        y1 = int(round((y_orig + h_orig) * scale))
        
        target_w = x1 - x0
        target_h = y1 - y0

        # Downsample the image
        # cv2.INTER_AREA is the best interpolation method for shrinking/downsampling images
        img_resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)

        # Paste the downsampled image into the scaled location on the canvas
        # This array slicing performs the exact same mathematical operation
        # as cv2.warpPerspective with the scaled Homography translation matrix!
        canvas[y0:y1, x0:x1] = img_resized
        
        log.info(f"Pasted {Path(img_path).name} mapped to ->[x={x0}, y={y0}, w={target_w}, h={target_h}]")

    # 4. Save result
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)
    log.info(f"Stitching complete. Saved downsampled whole slide to: {output_path}")

if __name__ == "__main__":
    # Adjust paths to match your directory structure
    META_PATH = Path("Cervical/PathPresenter/grid_renders/grid_metadata.json")
    OUTPUT_PATH = Path("Cervical/PathPresenter/grid_renders/whole_slide_10percent.png")
   
    if META_PATH.exists():
        # Downsample by 10%
        stitch_downsampled(META_PATH, OUTPUT_PATH, scale=0.1)
    else:
        log.error(f"Metadata not found at {META_PATH}!")
