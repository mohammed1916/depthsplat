"""
Convert a nerfstudio/blender-format transforms.json dataset into the
DepthSplat .torch chunk format so it can be fed to src.main inference.

Usage (standalone):
    python -m src.scripts.convert_photogrammetry \
        --data_dir  path/to/nerfstudio_data \
        --output_dir path/to/depthsplat_dataset \
        --scene_key  my_scene \
        --num_context_views 2

The script writes:
    <output_dir>/test/000000.torch   – single chunk containing the scene
    <output_dir>/test/index.json     – maps scene_key -> "000000.torch"
    <output_dir>/evaluation_index.json  – maps scene_key -> {context, target}

Camera format stored in .torch:
    cameras: Float32[N, 18]  =  [fx/w, fy/h, cx/w, cy/h, 0.0, 0.0,
                                  w2c_opencv[:3].flatten()]   (12 values)
    where w2c_opencv = inv(blender_c2w @ blender2opencv)
"""

from __future__ import annotations

import argparse
import json
import math
from io import BytesIO
from pathlib import Path

import numpy as np
import torch

try:
    from PIL import Image as PILImage
except ImportError:
    PILImage = None  # type: ignore


# Blender -> OpenCV coordinate change
BLENDER2OPENCV = np.array(
    [[1, 0, 0, 0],
     [0, -1, 0, 0],
     [0, 0, -1, 0],
     [0, 0, 0, 1]], dtype=np.float64
)


def _load_raw_jpeg(path: Path) -> torch.Tensor:
    """Load image file as raw bytes tensor (uint8). Re-encode to JPEG if not already JPEG."""
    raw = path.read_bytes()
    # Quick JPEG magic-number check
    if raw[:2] == b"\xff\xd8":
        return torch.tensor(list(raw), dtype=torch.uint8)
    # Re-encode to JPEG so the loader can decode with PIL
    if PILImage is None:
        raise RuntimeError("Pillow is required for non-JPEG images")
    buf = BytesIO()
    img = PILImage.open(path).convert("RGB")
    img.save(buf, format="JPEG", quality=95)
    raw = buf.getvalue()
    return torch.tensor(list(raw), dtype=torch.uint8)


def convert(
    data_dir: Path,
    output_dir: Path,
    scene_key: str,
    num_context_views: int = 2,
    target_image_hw: tuple[int, int] | None = None,
) -> dict:
    """
    Convert a nerfstudio data directory to a DepthSplat .torch chunk.

    Returns a dict with paths to the generated files.
    """
    transforms_path = data_dir / "transforms.json"
    if not transforms_path.is_file():
        raise FileNotFoundError(f"transforms.json not found in {data_dir}")

    meta = json.loads(transforms_path.read_text(encoding="utf-8"))

    w: int = meta["w"]
    h: int = meta["h"]
    fx: float = meta["fl_x"]
    fy: float = meta["fl_y"]
    cx: float = meta["cx"]
    cy: float = meta["cy"]

    # Normalised intrinsics (depthsplat convention)
    saved_fx = fx / w
    saved_fy = fy / h
    saved_cx = cx / w
    saved_cy = cy / h

    frames = meta["frames"]
    if not frames:
        raise ValueError("transforms.json contains no frames")

    timestamps: list[int] = []
    cameras_list: list[list[float]] = []
    images: list[torch.Tensor] = []

    for idx, frame in enumerate(frames):
        # file_path may be relative to data_dir or absolute
        fp = Path(frame["file_path"])
        if not fp.is_absolute():
            fp = data_dir / fp
        if not fp.is_file():
            # try without leading "./"
            fp = data_dir / Path(frame["file_path"]).name
        if not fp.is_file():
            raise FileNotFoundError(f"Image not found: {frame['file_path']} (checked {fp})")

        timestamps.append(idx)

        # transform_matrix is blender c2w -> convert to opencv w2c
        blender_c2w = np.array(frame["transform_matrix"], dtype=np.float64)
        opencv_c2w = blender_c2w @ BLENDER2OPENCV
        w2c = np.linalg.inv(opencv_c2w)          # 4x4 world-to-cam
        cam_row = [saved_fx, saved_fy, saved_cx, saved_cy, 0.0, 0.0]
        cam_row.extend(w2c[:3].flatten().tolist())  # 12 values
        cameras_list.append(cam_row)

        images.append(_load_raw_jpeg(fp))

    timestamps_t = torch.tensor(timestamps, dtype=torch.int64)
    cameras_t = torch.tensor(cameras_list, dtype=torch.float32)

    example = {
        "key": scene_key,
        "url": scene_key,
        "timestamps": timestamps_t,
        "cameras": cameras_t,
        "images": images,
    }

    # Write chunk
    stage_dir = output_dir / "test"
    stage_dir.mkdir(parents=True, exist_ok=True)
    chunk_path = stage_dir / "000000.torch"
    torch.save([example], chunk_path)

    # index.json
    index = {scene_key: "000000.torch"}
    (stage_dir / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")

    # evaluation_index.json  – pick evenly-spaced context frames; targets = all frames
    n = len(timestamps)
    num_ctx = min(num_context_views, n)
    if num_ctx < 2:
        ctx = list(range(n))
    else:
        step = max(1, (n - 1) // (num_ctx - 1))
        ctx = [min(i * step, n - 1) for i in range(num_ctx)]
        # deduplicate while preserving order
        seen: set[int] = set()
        ctx = [x for x in ctx if not (x in seen or seen.add(x))]  # type: ignore[func-returns-value]

    tgt = list(range(n))
    eval_index = {scene_key: {"context": ctx, "target": tgt}}
    eval_index_path = output_dir / "evaluation_index.json"
    eval_index_path.write_text(json.dumps(eval_index, indent=2), encoding="utf-8")

    # Determine actual image resolution from first frame
    actual_h, actual_w = h, w
    if PILImage is not None:
        try:
            fp0 = data_dir / Path(frames[0]["file_path"])
            with PILImage.open(fp0) as im:
                actual_w, actual_h = im.size
        except Exception:
            pass

    return {
        "chunk": str(chunk_path),
        "index_json": str(stage_dir / "index.json"),
        "evaluation_index_json": str(eval_index_path),
        "scene_key": scene_key,
        "num_frames": n,
        "num_context_views": num_ctx,
        "image_hw": [actual_h, actual_w],
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert nerfstudio transforms.json to DepthSplat .torch chunks"
    )
    p.add_argument("--data_dir", required=True, help="Directory containing transforms.json and images/")
    p.add_argument("--output_dir", required=True, help="Output directory for .torch chunks")
    p.add_argument("--scene_key", default="photogrammetry_scene", help="Scene identifier key")
    p.add_argument("--num_context_views", type=int, default=2, help="Number of context views for evaluation index")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = convert(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        scene_key=args.scene_key,
        num_context_views=args.num_context_views,
    )
    print(json.dumps(result, indent=2))
