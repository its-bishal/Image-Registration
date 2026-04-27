

import json
import logging
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s │ %(message)s")

cv2.ocl.setUseOpenCL(False)


class TileRenderer:
    """
    Lazy tile renderer backed by a registry JSON file.

    Parameters:
    registry_path : str | Path
        Path to the registry.json produced by register_tiles.py.
    blend : bool
        If True, use simple average blending in overlap regions.
        If False, last-write-wins (faster, slight seams).
    """

    def __init__(self, registry_path: str | Path, blend: bool = True):
        self.registry_path = Path(registry_path)
        self.blend = blend
        self._load_registry()

    def _load_registry(self):
        with open(self.registry_path) as f:
            reg = json.load(f)
        self.canvas_w = reg["canvas_width"]
        self.canvas_h = reg["canvas_height"]
        self.tiles = reg["tiles"]
        log.info(f"Registry loaded: {len(self.tiles)} tiles, "
                 f"canvas {self.canvas_w}×{self.canvas_h}")

    @staticmethod
    def _overlaps(bbox, rx, ry, rw, rh) -> bool:
        """True if tile bbox overlaps the requested rectangle."""
        xmin, ymin, xmax, ymax = bbox
        return not (xmax < rx or xmin > rx + rw or
                    ymax < ry or ymin > ry + rh)

    def render(self,
               x: int, y: int,
               width: int, height: int,
               output_path: str | None = None) -> np.ndarray:
        """
        Render a rectangular region of the virtual canvas.

        Parameters:
        x, y : top-left corner in canvas coordinates
        width, height : size of the requested region in pixels
        output_path : if given, also save the result to this path

        Returns
        np.ndarray BGR image of shape (height, width, 3)
        """
        # clamp to canvas
        x = max(0, x)
        y = max(0, y)
        width = min(width,  self.canvas_w - x)
        height = min(height, self.canvas_h - y)

        if width <= 0 or height <= 0:
            raise ValueError(" Requested region is outside the canvas.")

        log.info(f" Rendering region x={x} y={y} w={width} h={height}")

        # output buffers
        canvas = np.zeros((height, width, 3), dtype=np.float32)
        weight_map = np.zeros((height, width), dtype=np.float32)

        # translation that maps canvas coords → local output coords
        T_inv = np.array([[1, 0, -x],
                          [0, 1, -y],
                          [0, 0,  1]], dtype=np.float64)

        relevant = [t for t in self.tiles
                    if self._overlaps(t["bbox"], x, y, width, height)]
        log.info(f" Tiles overlapping region: {len(relevant)}")

        for tile in relevant:
            img = cv2.imread(tile["path"])
            if img is None:
                log.warning(f" Cannot read {tile['path']}, skipping.")
                continue

            th, tw = img.shape[:2]
            H_global = np.array(tile["H"], dtype=np.float64)

            H_local = T_inv @ H_global

            warped = cv2.warpPerspective(
                img.astype(np.float32),
                H_local,
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )

            # mask: pixels that the tile actually covers
            tile_mask = np.ones((th, tw), dtype=np.float32)
            warped_mask = cv2.warpPerspective(
                tile_mask, H_local, (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0,
            )

            if self.blend:
                canvas += warped * warped_mask[..., np.newaxis]
                weight_map += warped_mask
            else:
                m = warped_mask > 0.5
                canvas[m] = warped[m]
                weight_map[m] = 1.0

            del img, warped, warped_mask

        # normalise blended result
        if self.blend:
            mask = weight_map > 0
            canvas[mask] = canvas[mask] / weight_map[mask, np.newaxis]

        result = np.clip(canvas, 0, 255).astype(np.uint8)

        if output_path:
            cv2.imwrite(output_path, result)
            log.info(f" Saved => {output_path}")

        return result

    def render_full(self, output_path: str | None = None) -> np.ndarray:
        """Render the entire canvas. Practical only for small virtual canvases."""
        return self.render(0, 0, self.canvas_w, self.canvas_h, output_path)
    
    def render_bbox_coords(self,
                       coord1: tuple[int, int],
                       coord2: tuple[int, int],
                       output_path: str | None = None) -> np.ndarray:
        """
        Render the rectangle defined by two diagonal corner coordinates.

        Parameters:
        coord1 : (x1, y1) - one corner of the rectangle (canvas pixels)
        coord2 : (x2, y2) - the opposite diagonal corner
        output_path : optional save path

        Works regardless of which corner is top-left / bottom-right.
        """
        x1, y1 = coord1
        x2, y2 = coord2

        x = min(x1, x2)
        y = min(y1, y2)
        width = abs(x2 - x1)
        height = abs(y2 - y1)

        log.info(f"Diagonal coords: bbox: x={x} y={y} w={width} h={height}")
        return self.render(x, y, width, height, output_path)

    def info(self):
        print(f"Canvas : {self.canvas_w} x {self.canvas_h} px")
        print(f"Tiles : {len(self.tiles)}")
        for t in self.tiles[:5]:
            print(f" [{t['index']:4d}] r{t['row']:03d}_c{t['col']:03d}  "
                  f"bbox=({t['bbox'][0]:.0f},{t['bbox'][1]:.0f},"
                  f"{t['bbox'][2]:.0f},{t['bbox'][3]:.0f})")
        if len(self.tiles) > 5:
            print(f" and {len(self.tiles)-5} more")


if __name__ == "__main__":
    import sys

    REGISTRY = Path("Include/Cervical/PathPresenter/registry.json")
    OUT_DIR = Path("Include/Cervical/PathPresenter/renders")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not REGISTRY.exists():
        print(f"Registry not found: {REGISTRY}")
        print("Run register_tiles.py first.")
        sys.exit(1)

    renderer = TileRenderer(REGISTRY, blend=True)
    renderer.info()

    # renderer.render(
    #     x=0, y=0, width=3000, height=3000,
    #     output_path=str(OUT_DIR / "top_left_image.png")
    # )

    print("\nAll renders complete.")
    output_path= str(OUT_DIR/"cropped.png")

    img= renderer.render_bbox_coords(
        coord1=(5000, 100),
        coord2=(5800, 600),
        output_path= output_path
    )
