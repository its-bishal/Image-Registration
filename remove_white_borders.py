

import sys
from pathlib import Path
import numpy as np
from PIL import Image

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def find_crop_bounds(arr: np.ndarray, threshold: int, padding: int) -> tuple[int, int, int, int]:
    h, w = arr.shape[:2]
    rgb = arr[:, :, :3]
    
    # Identify content: pixels where at least one channel is below the threshold
    is_content = (rgb < threshold).any(axis=2) 

    row_has_content = is_content.any(axis=1)
    col_has_content = is_content.any(axis=0)

    if not row_has_content.any():
        return 0, 0, w, h

    top = int(np.argmax(row_has_content))
    bottom = int(h - np.argmax(row_has_content[::-1]))
    left = int(np.argmax(col_has_content))
    right = int(w - np.argmax(col_has_content[::-1]))

    top = max(0, top - padding)
    left = max(0, left - padding)
    bottom = min(h, bottom + padding)
    right = min(w, right + padding)

    return left, top, right, bottom


def remove_borders(src_path: Path, dst_path: Path, threshold: int = 245, padding: int = 0):
    img = Image.open(src_path)
    original_size = img.size

    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")

    arr = np.array(img)
    left, top, right, bottom = find_crop_bounds(arr, threshold, padding)

    cropped = img.crop((left, top, right, bottom))
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    cropped.save(dst_path)

    return original_size, cropped.size


def collect_images(paths: list[str]) -> list[Path]:
    images = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            for ext in SUPPORTED_EXTENSIONS:
                images.extend(path.glob(f"*{ext}"))
                images.extend(path.glob(f"*{ext.upper()}"))
        elif path.suffix.lower() in SUPPORTED_EXTENSIONS:
            images.append(path)
    return sorted(set(images))


def main():
    input_paths = ["C:/Users/ASUS/Downloads/microscope_screenshots"]
    output_directory = Path("Include/Cervical/PathPresenter/cropped")
    threshold_val = 230
    padding_val = 0

    images = collect_images(input_paths)
    if not images:
        print("No supported images found. Check your 'input_paths'.")
        return

    print(f"Found {len(images)} image(s) | Threshold: {threshold_val} | Padding: {padding_val}\n")
    print(f"{'Source':<45} {'Original':>14} {'Cropped':>14}  {'Status'}")
    print("-" * 85)

    ok = fail = 0
    for src in images:
        dst = output_directory / src.name
        
        try:
            orig, crop = remove_borders(src, dst, threshold_val, padding_val)
            orig_str = f"{orig[0]}×{orig[1]}"
            crop_str = f"{crop[0]}×{crop[1]}"
            status = "=>" if orig != crop else "="
            print(f"{str(src.name):<45} {orig_str:>14} {crop_str:>14}  {status:>6}")
            ok += 1
        except Exception as exc:
            print(f"{str(src.name):<45}  ERROR: {exc}")
            fail += 1

    print("-" * 85)
    print(f"Done — {ok} processed, {fail} failed.")



if __name__ == "__main__":
    main()

