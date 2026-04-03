"""
DualVideoPreview – ComfyUI custom node
Fast MP4 encoding matching VHS performance:
- Frames written to temp PNGs in parallel threads
- Single ffmpeg call with image sequence input (much faster than pipe)
- H.265 with fallback to H.264
"""

import os
import hashlib
import subprocess
import threading
import numpy as np
from PIL import Image
import folder_paths

output_dir = folder_paths.get_output_directory()

# Check codec availability once at import time
def _available_codecs():
    try:
        r = subprocess.run(["ffmpeg", "-encoders", "-v", "quiet"],
                           capture_output=True, text=True)
        return r.stdout
    except FileNotFoundError:
        return ""

_CODEC_LIST = _available_codecs()
HAS_H265 = "libx265" in _CODEC_LIST
HAS_H264 = "libx264" in _CODEC_LIST


def _write_frame(args):
    """Write a single frame PNG to disk. Called from thread pool."""
    frame_np, path = args
    img = Image.fromarray(frame_np)
    img.save(path, format="PNG", compress_level=1)  # level 1 = fast, not smallest


def _frames_to_mp4(frames_tensor, fps: float, uid_extra: str = "") -> str:
    """
    Convert (B, H, W, C) float32 tensor → H.265/H.264 MP4.
    Uses parallel PNG writes + ffmpeg image-sequence input for maximum speed.
    Returns filename relative to output_dir.
    """
    import tempfile
    from concurrent.futures import ThreadPoolExecutor

    frames = frames_tensor.cpu().numpy()
    B, H, W, C = frames.shape

    # Unique output filename
    uid = hashlib.md5((str(frames.shape) + str(fps) + uid_extra).encode()).hexdigest()[:12]
    filename = f"dvp_{uid}.mp4"
    filepath = os.path.join(output_dir, filename)

    if os.path.exists(filepath):
        return filename  # cached – skip encoding

    # Even dimensions required by codecs
    W_enc = W - (W % 2)
    H_enc = H - (H % 2)

    # Convert all frames to uint8 up front (vectorised, fast)
    frames_u8 = (np.clip(frames, 0.0, 1.0) * 255).astype(np.uint8)
    if W_enc != W or H_enc != H:
        frames_u8 = frames_u8[:, :H_enc, :W_enc, :]

    with tempfile.TemporaryDirectory(prefix="dvp_") as tmpdir:
        # --- Write PNGs in parallel -------------------------------------------
        paths = [os.path.join(tmpdir, f"f{i:06d}.png") for i in range(B)]
        with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as pool:
            list(pool.map(_write_frame, zip(frames_u8, paths)))

        # --- Pick encoder -----------------------------------------------------
        if HAS_H265:
            vcodec = "libx265"
            codec_args = [
                "-tag:v", "hvc1",   # Safari/QuickTime compat
                "-crf", "20",
                "-preset", "ultrafast",  # fastest encode, still good quality
                "-x265-params", "log-level=error",
            ]
        elif HAS_H264:
            print("[DualVideoPreview] H.265 unavailable, using H.264")
            vcodec = "libx264"
            codec_args = ["-crf", "18", "-preset", "ultrafast"]
        else:
            raise RuntimeError("[DualVideoPreview] No supported video encoder found (need libx265 or libx264)")

        # --- Single ffmpeg call with glob input -------------------------------
        cmd = [
            "ffmpeg", "-y",
            "-r", str(fps),
            "-i", os.path.join(tmpdir, "f%06d.png"),
            "-vcodec", vcodec,
            *codec_args,
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-an",           # no audio
            filepath,
        ]

        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"[DualVideoPreview] ffmpeg failed:\n{result.stderr.decode()}"
            )

    return filename


class DualVideoPreview:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "video_1":  ("STRING", {"default": "", "tooltip": "Path to first video file"}),
                "video_2":  ("STRING", {"default": "", "tooltip": "Path to second video file"}),
                "frames_1": ("IMAGE",  {"tooltip": "Frame sequence for video 1 (B,H,W,C)"}),
                "frames_2": ("IMAGE",  {"tooltip": "Frame sequence for video 2 (B,H,W,C)"}),
                "label_1":  ("STRING", {"default": "Before"}),
                "label_2":  ("STRING", {"default": "After"}),
                "fps":      ("FLOAT",  {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.5}),
                "loop":     ("BOOLEAN",{"default": True}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "preview_videos"
    OUTPUT_NODE = True
    CATEGORY = "image/video"
    DESCRIPTION = "Preview two videos with a drag-to-compare slider. Encodes frames as H.265 MP4 using parallel PNG writes for speed."

    def _resolve(self, path, frames, label, slot, fps):
        if frames is not None:
            try:
                filename = _frames_to_mp4(frames, fps, label + str(slot))
            except RuntimeError as e:
                print(e)
                return None
            return {"filename": filename, "subfolder": "", "type": "output",
                    "slot": slot, "label": label}

        if path and os.path.isfile(path):
            abs_out  = os.path.abspath(output_dir)
            abs_path = os.path.abspath(path)
            if abs_path.startswith(abs_out):
                rel = os.path.relpath(abs_path, abs_out)
                return {"filename": os.path.basename(rel),
                        "subfolder": os.path.dirname(rel),
                        "type": "output", "slot": slot, "label": label}
            return {"filename": os.path.basename(path), "subfolder": "",
                    "type": "output", "slot": slot, "label": label}

        return None

    def preview_videos(self, video_1="", video_2="", frames_1=None, frames_2=None,
                       label_1="Before", label_2="After", fps=24.0, loop=True):
        results = []
        # Encode both videos in parallel threads
        v1_result, v2_result = [None], [None]

        def enc1():
            v1_result[0] = self._resolve(video_1, frames_1, label_1, 1, fps)
        def enc2():
            v2_result[0] = self._resolve(video_2, frames_2, label_2, 2, fps)

        t1 = threading.Thread(target=enc1)
        t2 = threading.Thread(target=enc2)
        t1.start(); t2.start()
        t1.join();  t2.join()

        if v1_result[0]: results.append(v1_result[0])
        if v2_result[0]: results.append(v2_result[0])
        return {"ui": {"dual_videos": results, "loop": [loop]}}


NODE_CLASS_MAPPINGS       = {"DualVideoPreview": DualVideoPreview}
NODE_DISPLAY_NAME_MAPPINGS = {"DualVideoPreview": "Dual Video Preview 🎬"}
