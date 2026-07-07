from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cv2


VIDEOS = [
    Path("assets/demo_sofa_768.mp4"),
    Path("assets/demo_chair_768.mp4"),
]

BADGE_LINES = [
    "Robot: wheeled dual-arm mobile manipulator",
    "Navigation-only demo: wheeled base active, arms fixed",
]


def has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def add_badge(video_path: Path) -> None:
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    backup = video_path.with_name(video_path.stem + ".orig.mp4")
    if not backup.exists():
        shutil.copy2(video_path, backup)
        print(f"[backup] {backup}")
    else:
        print(f"[backup exists] {backup}")

    cap = cv2.VideoCapture(str(backup))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {backup}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 8.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tmp = video_path.with_name(video_path.stem + ".badged_tmp.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(tmp), fourcc, fps, (width, height))

    if not writer.isOpened():
        raise RuntimeError(f"Cannot open writer: {tmp}")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Semi-transparent dark panel.
        overlay = frame.copy()
        panel_h = 62
        cv2.rectangle(overlay, (0, 0), (width, panel_h), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.45, frame, 0.55, 0)

        y = 24
        for line in BADGE_LINES:
            cv2.putText(
                frame,
                line,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            y += 26

        writer.write(frame)

    cap.release()
    writer.release()

    if has_ffmpeg():
        final_tmp = video_path.with_name(video_path.stem + ".badged_h264.mp4")
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(tmp),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "28",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(final_tmp),
        ]
        subprocess.run(cmd, check=True)
        tmp.unlink(missing_ok=True)
        final_tmp.replace(video_path)
    else:
        tmp.replace(video_path)

    print(f"[ok] updated {video_path}")


def main():
    for video in VIDEOS:
        add_badge(video)

    # Keep GitHub Pages assets in sync.
    docs_assets = Path("docs/assets")
    docs_assets.mkdir(parents=True, exist_ok=True)

    for video in VIDEOS:
        dst = docs_assets / video.name
        shutil.copy2(video, dst)
        print(f"[sync] {video} -> {dst}")


if __name__ == "__main__":
    main()
