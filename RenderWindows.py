import json
import logging
from pathlib import Path
import numpy as np

from RenderRegion import TileRenderer

logging.basicConfig(level=logging.INFO, format="%(levelname)s │ %(message)s")
log = logging.getLogger(__name__)

def create_grid(registry_path: Path, output_dir: Path, rows: int = 5, cols: int = 7):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    renderer = TileRenderer(registry_path, blend=True)
    cw = renderer.canvas_w
    ch = renderer.canvas_h
    
    log.info(f"Partitioning canvas ({cw}x{ch}) into {rows} rows and {cols} columns.")
    
    x_edges = np.linspace(0, cw, cols + 1).astype(int)
    y_edges = np.linspace(0, ch, rows + 1).astype(int)
    
    cells_meta = []
    
    for r in range(rows):
        for c in range(cols):
            x0, x1 = x_edges[c], x_edges[c+1]
            y0, y1 = y_edges[r], y_edges[r+1]
            
            w = x1 - x0
            h = y1 - y0
            
            cell_filename = f"grid_r{r}_c{c}.png"
            cell_path = output_dir / cell_filename
            
            log.info(f"Rendering cell ({r}, {c}) at[x={x0}, y={y0}, w={w}, h={h}]")
            
            renderer.render(x=x0, y=y0, width=w, height=h, output_path=str(cell_path))
            
            cells_meta.append({
                "row": r,
                "col": c,
                "path": str(cell_path.resolve()),
                "bbox": [int(x0), int(y0), int(w), int(h)]
            })
            
    for cell in cells_meta:
        adjacencies = {}
        r, c = cell["row"], cell["col"]
        xA, yA, wA, hA = cell["bbox"]
        
        neighbors = {
            "top": (r - 1, c),
            "bottom": (r + 1, c),
            "left": (r, c - 1),
            "right": (r, c + 1)
        }
        
        for direction, (nr, nc) in neighbors.items():
            if 0 <= nr < rows and 0 <= nc < cols:
                neighbor_cell = next(n for n in cells_meta if n["row"] == nr and n["col"] == nc)
                xB, yB, wB, hB = neighbor_cell["bbox"]
                
                tx = xA - xB
                ty = yA - yB
                
                H = [
                    [1.0, 0.0, float(tx)],
                    [0.0, 1.0, float(ty)],
                    [0.0, 0.0, 1.0]
                ]    
                adjacencies[direction] ={
                    "target_row": nr,
                    "target_col": nc,
                    "H": H
                }
        cell["adjacencies"] = adjacencies
        
    metadata ={
        "canvas_width": cw,
        "canvas_height": ch,
        "grid_rows": rows,
        "grid_cols": cols,
        "cells": cells_meta
    }
    
    meta_path = output_dir / "grid_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
        
    log.info(f"Grid splitting complete. Metadata saved to {meta_path}")

if __name__ == "__main__":
    REGISTRY_FILE = Path("Include/Cervical/PathPresenter/registry.json")
    OUT_DIR = Path("Include/Cervical/PathPresenter/grid_renders")
    
    if not REGISTRY_FILE.exists():
        log.error(f"Cannot find {REGISTRY_FILE}")
    else:
        create_grid(REGISTRY_FILE, OUT_DIR, rows=5, cols=7)

