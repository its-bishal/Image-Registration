

import os
import re
import json
import logging
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s │ %(message)s")
log = logging.getLogger(__name__)

cv2.ocl.setUseOpenCL(False)

FOLDER_PATH = Path("PathPresenter/cropped")
REGISTRY_PATH = Path("PathPresenter/registry.json")
EXCLUDED_COLUMNS = {1, 2, 3, 33, 34, 35}
MAX_TILES = None  # set to None to use all
SIFT_NFEATURES = 1000
MATCH_RATIO = 0.75  # Lowe's ratio test
MIN_MATCH_COUNT = 10  # minimum good matches to accept a homography
WORK_MEGAPIX = 0.5  # downscale factor for feature extraction only
CHECKPOINT_EVERY = 50
CHECKPOINT_PATH = Path("PathPresenter/registry_checkpoint.json")


def custom_key(path: str):
    name = os.path.splitext(os.path.basename(path))[0]
    m = re.match(r"tile_r(\d+)_c(\d+)", name)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    return name


def collect_images(folder: Path) -> list[str]:
    def _not_excluded(p):
        m = re.match(r"tile_r(\d+)_c(\d+)",
                     os.path.splitext(os.path.basename(p))[0])
        return not m or int(m.group(2)) not in EXCLUDED_COLUMNS

    paths = sorted(
        [str(p) for p in folder.glob("*")
         if p.suffix.lower() in (".jpg", ".jpeg", ".png") and _not_excluded(p)],
        key=custom_key
    )
    return paths[:MAX_TILES] if MAX_TILES else paths


def load_for_features(path: str, work_megapix: float):
    """Load image, downscale for feature extraction, return (img_small, scale)."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {path}")
    h, w = img.shape[:2]
    area = h * w
    target_area = work_megapix * 1e6
    scale = min(1.0, (target_area / area) ** 0.5)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)
    return img, scale


def extract_features(sift, path: str, work_megapix: float):
    """Return (keypoints_full_res, descriptors, orig_wh) for one image."""
    img_small, scale = load_for_features(path, work_megapix)
    gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
    kps, descs = sift.detectAndCompute(gray, None)
    if kps is None or len(kps) == 0:
        return [], None, None

    # scale keypoints back to full-resolution coordinates
    for kp in kps:
        kp.pt = (kp.pt[0] / scale, kp.pt[1] / scale)
        kp.size /= scale

    # read original size without loading full image into RAM
    orig = cv2.imread(path)
    oh, ow = orig.shape[:2]
    del orig
    return kps, descs, (ow, oh)


def match_features(descs_a, descs_b, ratio=MATCH_RATIO):
    bf = cv2.BFMatcher(cv2.NORM_L2)
    raw = bf.knnMatch(descs_a, descs_b, k=2)
    return [m for m, n in raw if m.distance < ratio * n.distance]


def is_adjacent(tile_a: dict, tile_b: dict, max_step: int = 3) -> bool:
    """
    Return True if two tiles are close enough in (row, col) space to plausibly
    share visual overlap.  max_step=3 gives a small buffer for skipped tiles.
    """
    dr = abs(tile_b["row"] - tile_a["row"])
    dc = abs(tile_b["col"] - tile_a["col"])
    return dr <= max_step and dc <= max_step


def estimate_homography(kps_a, kps_b, good_matches):
    if len(good_matches) < MIN_MATCH_COUNT:
        return None
    src = np.float32([kps_a[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst = np.float32([kps_b[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    inliers = int(mask.sum()) if mask is not None else 0
    log.debug(f" homography inliers: {inliers}/{len(good_matches)}")
    return H if (H is not None and inliers >= MIN_MATCH_COUNT // 2) else None


def corners_of(w: int, h: int) -> np.ndarray:
    """4 corners of an image in homogeneous coordinates, shape (4,1,2)."""
    return np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)


def transform_corners(corners, H):
    return cv2.perspectiveTransform(corners, H).reshape(-1, 2)


def bbox_of(corners_2d: np.ndarray):
    xs, ys = corners_2d[:, 0], corners_2d[:, 1]
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


# checkpoint helpers
def save_checkpoint(tiles: list, failed: list, img_index: int):
    """Write a partial registry to disk immediately - no canvas normalisation yet."""
    if not tiles:
        return
    ckpt = {
        "status": "partial",
        "last_index": img_index,
        "tiles": tiles,
        "failed_indices": failed,
    }
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(CHECKPOINT_PATH) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(ckpt, f, indent=2)
    Path(tmp).replace(CHECKPOINT_PATH)
    log.info(f" Checkpoint saved: {CHECKPOINT_PATH}  ({len(tiles)} tiles)")
 
 
def load_checkpoint(image_paths: list[str]) -> tuple:
    """
    If a checkpoint exists, reload tiles + all_corners and return the index to
    resume from plus the previous tile's features so matching can continue.
    Returns (tiles, all_corners, failed, resume_idx, prev_kps, prev_descs, prev_H)
    or None if no checkpoint found.
    """
    if not CHECKPOINT_PATH.exists():
        return None
 
    with open(CHECKPOINT_PATH) as f:
        ckpt = json.load(f)
 
    if ckpt.get("status") != "partial":
        return None
 
    tiles = ckpt["tiles"]
    failed = ckpt.get("failed_indices", [])
    resume_idx = ckpt["last_index"] + 1
 
    log.info(f" Resuming from checkpoint: {len(tiles)} tiles already done, "
             f"continuing from index {resume_idx}")
 
    all_corners = [np.array(t["corners"]) for t in tiles]
 
    sift = cv2.SIFT_create(nfeatures=SIFT_NFEATURES)
    last_tile = tiles[-1]
    try:
        prev_kps, prev_descs, _ = extract_features(
            sift, last_tile["path"], WORK_MEGAPIX)
    except Exception as e:
        log.warning(f"Cannot re-extract features for last checkpoint tile: {e}")
        prev_kps, prev_descs = [], None
 
    prev_H = np.array(last_tile["H"], dtype=np.float64)
    return tiles, all_corners, failed, resume_idx, prev_kps, prev_descs, prev_H
 

def register(image_paths: list[str], resume: bool = True) -> dict:
    sift = cv2.SIFT_create(nfeatures=SIFT_NFEATURES)

    # attempt checkpoint resume
    resumed = load_checkpoint(image_paths) if resume else None

    if resumed:
        tiles, all_corners, failed, start_idx, prev_kps, prev_descs, prev_H = resumed
    else:

        tiles, all_corners, failed = [], [], []
        start_idx = 1

        log.info(f"Registering {len(image_paths)} tiles")
        p0 = image_paths[0]
        kps0, descs0, wh0 = extract_features(sift, p0, WORK_MEGAPIX)
        H0 = np.eye(3, dtype=np.float64)
        c0 = transform_corners(corners_of(*wh0), H0)
        b0 = bbox_of(c0)

        m = re.match(r"tile_r(\d+)_c(\d+)",
                    os.path.splitext(os.path.basename(p0))[0])
        tiles.append({
            "index": 0,
            "path": str(Path(p0).resolve()),
            "row": int(m.group(1)) if m else 0,
            "col": int(m.group(2)) if m else 0,
            "H": H0.tolist(),
            "corners": c0.tolist(),
            "bbox": list(b0),
        })
        all_corners.append(c0)

        prev_kps, prev_descs, prev_H = kps0, descs0, H0
    # failed = []

    for idx, path in enumerate(image_paths[start_idx:], start=start_idx):
        log.info(f" [{idx}/{len(image_paths)-1}] {Path(path).name}")
        try:
            m = re.match(r"tile_r(\d+)_c(\d+)",
                        os.path.splitext(os.path.basename(path))[0])

            try:
                kps, descs, wh = extract_features(sift, path, WORK_MEGAPIX)
            except Exception as e:
                log.warning(f" Cannot read tile, skipping: {e}")
                failed.append(idx)
                continue

            if descs is None or prev_descs is None:
                log.warning(" No descriptors, skipping.")
                failed.append(idx)
                continue

            # current_meta= {
            #     "row": int(m.group(1)) if m else idx,
            #     "col": int(m.group(2)) if m else 0
            # }
            # prev_meta= tiles[-1]

            # if not is_adjacent(prev_meta, current_meta):
            #     log.warning(f"  Non-adjascent jump."
            #                 f"skipping match using positional placement.")
            # else:
            good = match_features(prev_descs, descs)
            log.debug(f" Good matches: {len(good)}")

            # H_rel maps prev tile to current tile
            H_rel = estimate_homography(prev_kps, kps, good)

            if H_rel is None:
                log.warning(f"    Homography failed ({len(good)} matches) — "
                            "placing tile using identity offset.")
                # Fallback: assume simple translation from previous bbox
                if tiles:
                    prev_bbox = tiles[-1]["bbox"]
                    tx = prev_bbox[2] - prev_bbox[0]  # shift by prev tile width
                    H_rel = np.array([[1, 0, tx],
                                    [0, 1, 0],
                                    [0, 0, 1]], dtype=np.float64)
                else:
                    H_rel = np.eye(3)

            # Global: H_global = H_prev_global @ inv(H_rel)
            # Because H_rel maps coords in prev-tile space to current-tile 
            # space, and we need the inverse to go current-tile to prev-tile to canvas.
            H_rel_inv = np.linalg.inv(H_rel)
            H_global = prev_H @ H_rel_inv

            corners = transform_corners(corners_of(*wh), H_global)
            bbox = bbox_of(corners)
            all_corners.append(corners)

            tiles.append({
                "index": idx,
                "path": str(Path(path).resolve()),
                "row": int(m.group(1)) if m else idx,
                "col": int(m.group(2)) if m else 0,
                "H": H_global.tolist(),
                "corners": corners.tolist(),
                "bbox": list(bbox),
            })

            # slide window: current becomes previous
            prev_kps, prev_descs, prev_H = kps, descs, H_global

            # periodic checkpoint
            if CHECKPOINT_EVERY and (idx % CHECKPOINT_EVERY == 0):
                save_checkpoint(tiles, failed, idx)
        except Exception as _e:
            log.error(f"    Unexpected error on tile {idx}: {_e}")
            log.error(f"     Skipping tile and coninuing")
            failed.append(idx)
    
    # Saving final checkpoint before canvas normalization
    save_checkpoint(tiles, failed, len(image_paths) - 1)

    all_pts = np.vstack(all_corners)
    min_x = float(all_pts[:, 0].min())
    min_y = float(all_pts[:, 1].min())
    max_x = float(all_pts[:, 0].max())
    max_y = float(all_pts[:, 1].max())

    if min_x != 0 or min_y != 0:
        shift = np.array([[1, 0, -min_x],
                          [0, 1, -min_y],
                          [0, 0,  1]], dtype=np.float64)
        for t in tiles:
            H_new = shift @ np.array(t["H"])
            c_new = transform_corners(corners_of(
                               int(round(t["bbox"][2] - t["bbox"][0])),
                               int(round(t["bbox"][3] - t["bbox"][1]))), H_new)
            orig_w = cv2.imread(t["path"]).shape[1]
            orig_h = cv2.imread(t["path"]).shape[0]
            c_new = transform_corners(corners_of(orig_w, orig_h), H_new)
            t["H"] = H_new.tolist()
            t["corners"] = c_new.tolist()
            t["bbox"] = list(bbox_of(c_new))

        max_x -= min_x
        max_y -= min_y

    registry = {
        "canvas_width": int(round(max_x)),
        "canvas_height": int(round(max_y)),
        "tiles": tiles,
        "failed_indices": failed,
    }

    log.info(f"\nCanvas size : {registry['canvas_width']} x {registry['canvas_height']}")
    log.info(f"Tiles registered: {len(tiles)}  |  failed: {len(failed)}")
    return registry


if __name__ == "__main__":
    imgs = collect_images(FOLDER_PATH)
    log.info(f"Found {len(imgs)} tiles.")
    registry = register(imgs)

    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)
    log.info(f"Registry saved : {REGISTRY_PATH}")
